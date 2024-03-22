import re
import torch
import pickle
import numpy as np
from tqdm import tqdm
from util import *
from loss import *
from model import *
from pretrain import *
from dataloader import *
from init_parameter import *
from torch.nn.functional import softmax
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer, BertTokenizer

class ModelManager:
    
    def __init__(self, args, data, pretrained_model=None):
        self.args = args
        self.data = data
        self.model = pretrained_model

        if self.model is None:
            self.model = BertForModel.from_pretrained(args.bert_model, cache_dir = "", num_labels = data.num_labels)
            self.restore_model(args)

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id     
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
        
        self.best_eval_score = 0
        self.delta = None
        self.delta_points = []
        self.centroids = None

        self.test_results = None
        self.predictions = None
        self.true_labels = None

    def open_classify(self, features):

        logits = euclidean_metric(features, self.centroids)
        probs, preds = F.softmax(logits.detach(), dim = 1).max(dim = 1)
        euc_dis = torch.norm(features - self.centroids[preds], 2, 1).view(-1)
        preds[euc_dis >= self.delta[preds]] = data.unseen_token_id

        return preds

    def get_features_and_labels(model, dataloader, device):
        model.eval()
        all_features = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                
                input_ids, attention_masks, segment_ids, labels = [b.to(device) for b in batch]
                features = model(input_ids=input_ids, attention_mask=attention_masks)[1]

                all_features.extend(features.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())

        return np.array(all_features), np.array(all_labels)

    def evaluation(self, args, data, mode="eval"):
        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        if mode == 'eval':
            dataloader = data.eval_dataloader
        elif mode == 'test':
            dataloader = data.test_dataloader

        for batch in tqdm(dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                pooled_output, _ = self.model(input_ids, segment_ids, input_mask)
                preds = self.open_classify(pooled_output)

                total_labels = torch.cat((total_labels,label_ids))
                total_preds = torch.cat((total_preds, preds))
        
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        self.predictions = list([data.label_list[idx] for idx in y_pred])
        self.true_labels = list([data.label_list[idx] for idx in y_true])

        if mode == 'eval':
            cm = confusion_matrix(y_true, y_pred)
            plot_confusion_matrix(cm, data.all_label_list, "confusion_matrix.png", normalize=True)
            eval_score = F_measure(cm)['F1-score']
            return eval_score
            
        elif mode == 'test':
            
            cm = confusion_matrix(y_true,y_pred)
            results = F_measure(cm)
            acc = round(accuracy_score(y_true, y_pred) * 100, 2)
            results['Accuracy'] = acc

            self.test_results = results
            self.save_results(args)

    def train(self, args, data):     
        
        criterion_boundary = BoundaryLoss(num_labels = data.num_labels, feat_dim = args.feat_dim)
        self.delta = F.softplus(criterion_boundary.delta)
        optimizer = torch.optim.Adam(criterion_boundary.parameters(), lr = args.lr_boundary)
        self.centroids = self.centroids_cal(args, data)

        wait = 0
        best_delta, best_centroids = None, None

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(data.train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):
                    features = self.model(input_ids, segment_ids, input_mask, feature_ext=True)
                    loss, self.delta = criterion_boundary(features, self.centroids, label_ids)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    tr_loss += loss.item()
                    
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

            self.delta_points.append(self.delta)
            
            # if epoch <= 20:
            #     plot_curve(self.delta_points)

            loss = tr_loss / nb_tr_steps
            print('train_loss',loss)
            
            eval_score = self.evaluation(args, data, mode="eval")
            print('eval_score',eval_score)
            
            if eval_score >= self.best_eval_score:
                wait = 0
                self.best_eval_score = eval_score
                best_delta = self.delta
                best_centroids = self.centroids
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break
        
        self.delta = best_delta
        self.centroids = best_centroids

    def class_count(self, labels):
        class_data_num = []
        for l in np.unique(labels):
            num = len(labels[labels == l])
            class_data_num.append(num)
        return class_data_num

    def centroids_cal(self, args, data):
        centroids = torch.zeros(data.num_labels, args.feat_dim).cuda()
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)

        with torch.set_grad_enabled(False):
            for batch in data.train_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                features = self.model(input_ids, segment_ids, input_mask, feature_ext=True)
                total_labels = torch.cat((total_labels, label_ids))
                for i in range(len(label_ids)):
                    label = label_ids[i]
                    centroids[label] += features[i]
                
        total_labels = total_labels.cpu().numpy()
        centroids /= torch.tensor(self.class_count(total_labels)).float().unsqueeze(1).cuda()
        
        return centroids

    def restore_model(self, args):
        output_model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        self.model.load_state_dict(torch.load(output_model_file))
    
    def save_results(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        var = [args.dataset, args.known_cls_ratio, args.labeled_ratio, args.seed]
        names = ['dataset', 'known_cls_ratio', 'labeled_ratio', 'seed']
        vars_dict = {k:v for k,v in zip(names, var) }
        results = dict(self.test_results,**vars_dict)
        keys = list(results.keys())
        values = list(results.values())
        
        np.save(os.path.join(args.save_results_path, 'centroids.npy'), self.centroids.detach().cpu().numpy())
        np.save(os.path.join(args.save_results_path, 'deltas.npy'), self.delta.detach().cpu().numpy())

        file_name = 'results.csv'
        results_path = os.path.join(args.save_results_path, file_name)
        
        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori,columns = keys)
            df1.to_csv(results_path,index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results,index=[1])
            df1 = df1.append(new,ignore_index=True)
            df1.to_csv(results_path,index=False)
        data_diagram = pd.read_csv(results_path)
        
        print('test_results', data_diagram)
    
    def save_model(self, file_path='adb_model.pkl'):
        with open(file_path, 'wb') as file:
            pickle.dump(self.model, file)
        print(f'Model saved to {file_path}')

    def load_model(file_path='adb_model.pkl'):
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model

    def preprocess_input(self, text, tokenizer, max_length=512):
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
        return inputs

    def classify_user_input(self, text):
        self.model.eval()
        inputs = self.preprocess_input(text, self.tokenizer,self.args.max_seq_length )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            probs = softmax(logits, dim=-1)
            predicted_index = torch.argmax(probs, dim=-1).cpu().numpy()
            predicted_label = [self.data.all_label_list[idx] for idx in predicted_index][0]
        return predicted_label

if __name__ == '__main__':
    
    print('Data and Parameters Initialization...')
    parser = init_model()
    args = parser.parse_args()
    data = Data(args)

    print('Pre-training begin...')
    manager_p = PretrainModelManager(args, data)
    manager_p.train(args, data)
    print('Pre-training finished!')
    
    manager = ModelManager(args, data, manager_p.model)
    print('Training begin...')
    manager.train(args, data)
    print('Training finished!')
    
    print('Evaluation begin...')
    manager.evaluation(args, data, mode="test")  
    print('Evaluation finished!')

    manager.save_model('/content/home/drive/MyDrive/Adaptive-Decision-Boundary/adb_model.pkl')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    debug(data, manager_p, manager, args)

    # Load the model
    model_path = "/content/home/drive/MyDrive/Adaptive-Decision-Boundary/adb_model.pkl"
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    model.to(device)
    model.eval()
    tokenizer = manager.tokenizer

    user_input = input("Enter your query: ")
    predicted_label = manager.classify_user_input(user_input)
    print(f"Predicted Label: {predicted_label}")
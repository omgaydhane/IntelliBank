
# New Intent Detection and Discovery

Modern dialogue systems have made significant strides in understanding and responding to user queries. Despite these advancements, they still encounter challenges when dealing with diverse user intents, especially those not covered in their training data. This limitation often leads to inaccurate responses and misinterpretations, impeding the potential for meaningful interaction.

To overcome these limitations, we introduce a cutting-edge methodology that combines:

1. **Adaptive Decision Boundary (ADB) Model:** For precise open intent detection.
2. **Multi-Task Pre-training and Contrastive Learning with Nearest Neighbors (MTP-CLNN):** For efficient intent discovery.

## ADBES and MTP-CLNN Integration

ADB’s key feature is its ability to dynamically adjust decision boundaries surrounding known intent clusters, effectively identifying open intents that fall outside these clusters. To further enhance the model's accuracy, we have upgraded the ADB model to Adaptive Decision Boundary Learning via Expanding and Shrinking (ADBES). This introduces a new loss function, allowing for a more precise encapsulation of known intents and better differentiation from emerging or unknown ones.

Through the combination of ADBES and MTP-CLNN, our pipeline:

- Accurately identifies known intents.
- Uncovers new intent categories.
- Facilitates a robust and adaptable dialogue system capable of evolving with user needs.

### Application: IntelliBank

The development of IntelliBank, a service bot designed for open intent recognition in banking conversations, represents a significant application of our findings. By integrating ADBES for precise open intent detection and MTP-CLNN for open intent discovery, IntelliBank:

- Substantially improves intent recognition accuracy.
- Effectively discerns user queries.
- Suggests relevant keywords upon detecting open intents, facilitating efficient query resolution by bank staff and enhancing customer service.

## Key Features

1. **Framework Overview:** Introduces a framework for identifying established intents and discovering new, open intents within datasets. Structured around two core modules: Open Intent Detection and Open Intent Discovery.

2. **Open Intent Detection Module:** Utilizes deep learning algorithms to categorize known intents with high accuracy, applying them to unlabeled data to identify known intents and potential open intents.

3. **Open Intent Discovery Module:** Augments training data with both recognized known intents and newly identified open intents, applying user-selected clustering techniques to identify distinct clusters of open intents.

4. **Integrating Keyword-based Labeling:** Extracts pivotal keywords from sentences within open intent clusters, assigning informative, keyword-based labels to enhance dataset understanding.

5. **Adaptive Decision Boundary (ADB) Technique:** Addresses classifying open intents dynamically, defining decision boundaries using central points and variable radii, and utilizing a loss function to adjust decision boundaries based on proximity and class alignment.

6. **Loss Function and Boundary Adjustment Mechanism:** Engineered to balance expansion and contraction of decision boundaries, consisting of positive and negative loss components for adjusting boundaries, with a nuanced scaling mechanism to moderate the impact of negative loss.

## Installation

### Project Directory
```
Project Directory
│
├── ADBES
│   ├── ADB.py               # Code implementing the ADBES model.
│   ├── confusion_matrix.png # Visualization showing the model's performance.
│   ├── dataloader.py        # Functions for data loading and preprocessing.
│   ├── init_parameter.py    # Initialization settings for the model.
│   ├── loss.py              # Implementation of the loss function for model training.
│   ├── model.py             # Main model architecture code.
│   ├── pretrain.py          # Code for pre-training the model.
│   ├── util.py              # Utility functions for the ADBES model.
│   ├── requirements.txt     # Dependencies for the ADBES model.
│   ├── data                 # Directory containing datasets used for training.
│   ├── figs                 # Directory for figures and visualizations.
│   ├── outputs              # Directory for output files from the model.
│   └── results              # Directory for results from model evaluations.
│
├── MTP-CLNN
│   ├── clnn.py              # Code implementing the CLNN model.
│   ├── dataloader.py        # Functions for data loading and preprocessing.
│   ├── init_parameter.py    # Initialization settings for the model.
│   ├── model.py             # Main model architecture code.
│   ├── mtp.py               # Code for managing multi-task pre-training.
│   ├── get-pip.py           # Script for installing necessary packages.
│   ├── clnn_outputs         # Directory for output files from the model.
│   ├── data/banking         # Directory for datasets for training.
│   ├── images               # Directory for visualizations and figures.
│   ├── scripts              # Additional scripts for MTP-CLNN.
│   └── utils                # Utility functions for the MTP-CLNN model.
│
├── IntelliBank
│   ├── manage.py            # Main script for managing the Django-based application.
│   ├── db.sqlite3           # SQLite database storing application data.
│   ├── Dockerfile           # Configuration for Docker setup.
│   ├── requirements.txt     # Dependencies for running the application.
│   ├── api                  # Backend code for API management.
│   ├── core                 # Core application code, including frontend and Django settings.
│   ├── static               # Static files including CSS and JavaScript.
│   ├── templates            # HTML templates for the application's frontend.
│   └── __pycache__          # Directory storing cached Python files.
```

## Contact Us

For inquiries or support, please contact us at [contact@newintentdiscovery.com](mailto:contact@newintentdiscovery.com).

New Intent Detection and Discovery

Modern dialogue systems have made significant strides in understanding and responding to user queries. Despite these advancements, they still encounter challenges when dealing with diverse user intents, especially those not covered in their training data. This limitation often leads to inaccurate responses and misinterpretations, impeding the potential for meaningful interaction. To overcome these limitations, we introduce a cutting-edge methodology that combines an enhanced Adaptive Decision Boundary (ADB) model for precise open intent detection with Multi-Task Pre-training and Contrastive Learning with Nearest Neighbors (MTP-CLNN) for efficient intent discovery.

ADB’s key feature is its ability to dynamically adjust decision boundaries surrounding known intent clusters, thereby effectively identifying open intents that fall outside these clusters. To further enhance the accuracy of the model, we have upgraded the ADB model to Adaptive Decision Boundary Learning via Expanding and Shrinking (ADBES). This enhancement includes an update to the model's loss function, by introducing the concept of shrinking boundaries. This modification allows for a more precise encapsulation of known intents and a better differentiation from emerging or unknown ones. Through the combination of ADBES and MTP-CLNN, our pipeline not only accurately identifies known intents but also uncovers new intent categories, facilitating a more robust and adaptable dialogue system capable of evolving with user needs.

The development of IntelliBank, a Service Bot designed for Open Intent Recognition in banking conversations, represents a significant application of our findings. By integrating ADBES for precise Open Intent Detection, alongside MTP-CLNN for Open Intent Discovery, IntelliBank substantially improves intent recognition accuracy. It effectively discerns user queries and, upon detecting open intents, suggests relevant keywords. This facilitates efficient query resolution by bank staff, optimizing operations and enhancing customer service.

Key Features:

1. Framework Overview:
    Introduces a framework for identifying established intents and discovering new, open intents within datasets.
    Structured around two core modules: Open Intent Detection and Open Intent Discovery.
    Preprocessing step ensures data quality and relevancy.

2.Open Intent Detection Module:
    Utilizes deep learning algorithms to categorize known intents with high accuracy.
    Applied to unlabeled data to identify known intents and potential open intents.

3.Open Intent Discovery Module:
    Augments training data with both recognized known intents and newly identified open intents.
    Applies user-selected clustering techniques to identify distinct clusters of open intents.

4.Integrating Keyword-based Labeling:
    Extracts pivotal keywords from sentences within open intent clusters.
    Assigns informative, keyword-based labels to enhance dataset understanding.

5.Adaptive Decision Boundary (ADB) Technique:
    Addresses classifying open intents dynamically.
    Defines decision boundaries using central points and variable radii.
    Utilizes a loss function to adjust decision boundaries based on proximity and class alignment.  

6.Loss Function and Boundary Adjustment Mechanism:
    Engineered to balance expansion and contraction of decision boundaries.
    Consists of positive and negative loss components for adjusting boundaries.
    Nuanced scaling mechanism moderates the impact of negative loss to prevent overly aggressive contraction.
    Installation


Contributing

We welcome contributions from the community! If you're interested in contributing to New Intent Detection and Discovery, please check out our Contribution Guidelines.

License

This project is licensed under the MIT License.

Contact Us

For inquiries or support, please contact us at contact@newintentdiscovery.com.

# A Survey of Human-object Interaction Detection with Deep Learning

> [A Survey of Human-object Interaction Detection with Deep Learning]() <br>
> [![paper](https://img.shields.io/badge/Paper-arxiv-b31b1b)](https://arxiv.org/pdf/2301.00394.pdf)

If you find this repository helpful, please consider citing:

```BibTeX
@article{yang2023humanparsing,
  title={Deep Learning Technique for Human Parsing: A Survey and Outlook},
  author={Lu Yang and Wenhe Jia and Shan Li and Qing Song},
  journal={arXiv preprint arXiv:2301.00394},
  year={2023}
}
```


## Contributing 

Please feel free to create issues or pull requests to add papers.

## 1. Introduction
Human-object interaction (HOI) detection has attracted significant attention due to its wide applications including human-robot interactions, security monitoring, automatic sports commentary, etc. HOI detection aims to detect humans, objects, and their interactions in a given image or video, so it needs a higher-level semantic understanding of the image than regular object recognition or detection tasks. It is also more challenging technically because of some unique difficulties, such as multiobject interactions and group activities, etc. Currently, deep learning methods have achieved great performance in HOI detection, but there are few reviews describing the recent advance of deep learning-based HOI detection. Moreover, the current stage-based category of HOI detection methods is causing confusion in community discussion and beginner learning. To fill this gap, this paper summarizes, categorizes, and compares HOI detection methods over the last decade. Firstly, we summarize the pipeline of HOI detection methods. Then, we divide existing methods into three categories (Instance feature-based, interaction region-based, and query-based), distinguish them in formulas and schematics, and qualitatively compare their advantages and disadvantages. After that, we review each category of methods in detail. We also quantitatively compare the performance of existing methods on public HOI datasets. At last, we point out the future research direction of HOI detection.


## 2.  Overview of HOI detection

Deep learning methods have achieved brilliant achievements in object recognition and detection, which greatly reduce manual labor in processing mass visual information. Object recognition aims to answer ``What is in the image", while object detection aims to answer ``What and where is in the image". However, an expected intelligent machine should have a complete semantics understanding of an scene. Towards this goal, human-object interaction (HOI) detection are proposed to answer ``What are the people doing with what objects?". Fig. 1 gives two examples to show the different goals between object recognition, object detection and HOI detection. From which we can see, HOI detection can provide more human-centered information in the semantics level. Therefore, HOI detection has plenty of application potential in human-robot interactions, security monitoring, automatic sport commentary, action simulation and recognition, etc. At the same time, HOI detection plays a crucial role in the embodied AI system, which thinks that human intelligence needs to be formed through interaction and iteration with actual scenes. 
<p align="center"><img width="90%" src="pics/fig_diff2.png" /></p>

Existing deep HOI detection methods follow a four-step pipeline, as shown in Fig. 2. Firstly, the model takes an image or a video as the main input. We aim to find out all the HOIs in the image or video. In addition to visual information, human body model has been used as prior knowledge to improve results. Text corpus has also been used as external clues to detect unseen objects or actions. Secondly, HOI detection methods utilize some off-the-shelf backbone networks to extract features from inputs. Generally, these backbone networks have been pre-trained on large-scale datasets, and their weights are frozen during HOI detection training. An excellent pre-training method can affect the final detection accuracy. Thirdly, the HOI predictor further learns HOI-specific features and then predicts the HOI triplets. The HOI predictor is the core of HOI algorithms, which could be based on various structures, such as CNN, LSTM, GCN, Transformer, etc. Finally, HOI detection model outputs the $\langle$human-verb-object$\rangle$ triplets existed in the image or video. 
<p align="center"><img width="90%" src="pics/fig_pipeline_all.png" /></p>

Although the stage-based category is causing confusions for discussion and beginners, it has already become the mainstream category in more than a hundred papers. Therefore, this survey tries to just rename them but clearly demonstrate the essential differences between the different categories. Specifically, we rename them two-stage, one-stage, and end-to-end methods into instance feature-based methods, interaction-region-based methods, and query-based methods. The guideline for our category is how each type of method represents the interaction.
<p align="center"><img width="90%" src="pics/fig_compare.png" /></p>

Fig. 3 the development process of HOI detection methods and HOI-related datasets. For clarity, Fig. 3 only parts but not all existing methods. The listed HOI methods meet the following two conditions: (1) solve a typical HOI detection problem; (2) can be clearly classified into one of the proposed three categories.
<p align="center"><img width="90%" src="pics/fig_all.png" /></p>

## 3. Instance feature-based methods
Instance feature-based methods use the appearance of detected instances (either humans or objects) as cues to predict the interaction between them. Therefore, the instance feature-based method generally consists of two sequential steps: instance detection and interaction classification. In the first stage, it uses an object detector, such as a Faster RCNN to detect the human and object instances. The output of the first stage includes the labels, bounding box, and in-box features of the detected instances. In the second stage, it uses features in the detected box to identify the interaction between each possible human-object pair. Note that the weights of the first-stage detector can be either fixed or updated during training.
<p align="center"><img width="90%" src="pics/fig_horcnn.png" /></p>
<p align="center"><img width="90%" src="pics/fig_two.png" /></p>


## 4. Interaction region-based methods
Interaction region-based methods aim to regress a region to represent the interaction. The interaction region could be a point, dynamic points, a union box or multi-scale boxes. In another word, these methods detect simultaneously detect human instances, object instances, and some interaction areas or points, where the interaction areas are only used to predict interaction verbs. Therefore, they usually follow a parallel structure.
<p align="center"><img width="90%" src="pics/fig_pipeline_one.png" /></p>
<p align="center"><img width="90%" src="pics/fig_one.png" /></p>


## 5. Query-based methods
Query-based methods use trainable query vectors to represent HOI triplets. Their basic architecture is a transformer encoder-decoder. The encoder uses an attention mechanism to extract features from the global image context. The decoder takes several learnable query vectors as input and each query captures at most one interaction action of a human-object pair. Actually, these methods just extend the transformer-based detection model DETR to capture HOI detection, and treat HOI detection as a set prediction problem of matching the predicted and ground-truth HOI instances. 
<p align="center"><img width="90%" src="pics/fig_pipeline_trans.png" /></p>
<p align="center"><img width="90%" src="pics/fig_transformebased_timeline.png" /></p>


## 6. Dataset, metrics and performance
In this section, we summarize the information of popular HOI detection datasets. From 2015 to 2022, a total of 15 datasets for HOI detection emerged, including HICO, V-COCO, HICO-DET, HCVRD, HOI-A, HAKE, Ambiguous-HOI, MECCANO, HOI-VP, VidHOI, V-HICO, SWiG-HOI, D3D-HOI, BEHAVE and HOI4D.
<p align="center"><img width="90%" src="pics/fig_dataset.png" /></p>


- [3.1 Single Human Parsing (SHP) Models](https://github.com/soeaver/awesome-human-parsing/blob/main/3-HP.md#31-Single-Human-Parsing-Models)
- [3.2 Multiple Human Parsing (MHP) Models](https://github.com/soeaver/awesome-human-parsing/blob/main/3-HP.md#32-Multiple-Human-Parsing-Models)
- [3.3 Video Human Parsing (VHP) Models](https://github.com/soeaver/awesome-human-parsing/blob/main/3-HP.md#33-Video-Human-Parsing-Models)



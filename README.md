# A Survey of Human-object Interaction Detection with Deep Learning

> [A Survey of Human-object Interaction Detection with Deep Learning]() <br>
> [![paper](A_Survey_of_Human-object_Interaction_Detection_with Deep_Learning.pdf)]

---

If you find this repository helpful, please consider citing:

```BibTeX
@ARTICLE{10816567,
  author={Han, Geng and Zhao, Jiachen and Zhang, Lele and Deng, Fang},
  journal={IEEE Transactions on Emerging Topics in Computational Intelligence}, 
  title={A Survey of Human-Object Interaction Detection With Deep Learning}, 
  year={2024},
  volume={},
  number={},
  pages={1-24},
  keywords={Transformers;Feature extraction;Reviews;Foundation models;Visualization;Object recognition;Pipelines;Surveys;Semantics;Heavily-tailed distribution;Deep learning;visual relationship detection;human-object interaction;foundation models;attention mechanism;GNN;transformer},
  doi={10.1109/TETCI.2024.3518613}}
```


---
## Contributing 

Compared with the currently published HOI detection review papers, our contributions can be summarized as follows:

(1) We review more than 200 references related to HOI detection and 13 datasets from 2015 to 2024, and compare the advantages and disadvantages of HOI detection methods and datasets. Then we summarize the pipeline of all three classes of HOI detection methods and clearly distinguish them in formulas and schematics.

(2) We analyze the impact of foundation models on HOI detection methods, which is not covered in the previous HOI field review.

(3) Based on the analyzed papers, we reasonably deduce and explore future research directions, analyze the current problems and limitations of each research direction, and propose our suggestions to solve these problems.

---
## Contents

  - [Overview of HOI detection](#Overview-of-HOI-detection)
  - [Two-stage methods](#Two-stage-methods)
  - [One-stage methods](#One-stage-methods)
  - [Transformer-based methods](#Transformer-based-methods)
  - [Foundation models methods](#Foundation-models-methods)

---
## Overview of HOI detection

Existing deep HOI detection methods follow a common four-step pipeline, as shown in Fig.1. Firstly, the model takes an image as the main input. Secondly, HOI detection methods utilize some off-the-shelf backbone networks to extract features from inputs. Thirdly, the HOI predictor further learns HOI-specific features and then predicts the HOI triplets. Finally, HOI detection model outputs the human-verb-object triplets existed in the image. 
<p align="center"><img width="90%" src="pics/fig_pipeline_all.png" /></p>
Fig. 1 Pipeline of HOI detection using deep learning methods.

<p align="center"><img width="90%" src="pics/fig_all.png" /></p>
Fig. 2 The development process of HOI detection methods. 

---
<details>
<summary>## Two-stage methods</summary>

- [73] HO-RCNN:Learning to detect human-object interactions[[Paper]](https://ieeexplore.ieee.org/abstract/document/8354152)
- [27] ICAN:iCAN: Instance-Centric Attention Network for Human-Object Interaction Detection[[Paper]](https://arxiv.org/abs/1808.10437)
- [74] GPNN:Learning human-object interactions by graph parsing neural networks[[Paper]](https://openaccess.thecvf.com/content_ECCV_2018/html/Siyuan_Qi_Learning_Human-Object_Interactions_ECCV_2018_paper.html)
- [32] InteractNet:Detecting and recognizing human-object interactions[[Paper]](https://openaccess.thecvf.com/content_cvpr_2018/html/Gkioxari_Detecting_and_Recognizing_CVPR_2018_paper.html)
- [95] Pairwise:Pairwise body-part attention for recognizing human-object interactions[[Paper]](https://openaccess.thecvf.com/content_ECCV_2018/html/Haoshu_Fang_Pairwise_Body-Part_Attention_ECCV_2018_paper.html)
- [91] Context:Deep contextual attention for human-object interaction detection[[Paper]](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Deep_Contextual_Attention_for_Human-Object_Interaction_Detection_ICCV_2019_paper.html)
- [62] PMF-Net:Pose-aware multi-level feature network for human object interaction detection[[Paper]](https://openaccess.thecvf.com/content_ICCV_2019/html/Wan_Pose-Aware_Multi-Level_Feature_Network_for_Human_Object_Interaction_Detection_ICCV_2019_paper.html)
- [28] RPNN:Relation parsing neural network for human-object interaction detection[[Paper]](https://openaccess.thecvf.com/content_ICCV_2019/html/Zhou_Relation_Parsing_Neural_Network_for_Human-Object_Interaction_Detection_ICCV_2019_paper.html)
- [85] BAR-CNN:Detecting visual relationships using box attention[[Paper]](https://openaccess.thecvf.com/content_ICCVW_2019/html/SGRL/Kolesnikov_Detecting_Visual_Relationships_Using_Box_Attention_ICCVW_2019_paper.html?ref=https://githubhelp.com)
- [67] Turbo:Turbo learning framework for human-object interactions recognition and human pose estimation[[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/3878)
- [61] PMN:Pose-based modular network for human-object interaction detection[[Paper]](https://arxiv.org/abs/2008.02042)
- [29] FCMNet:Amplifying key cues for human-object-interaction detection[[Paper]](https://link.springer.com/chapter/10.1007/978-3-030-58568-6_15)
- [89] Cascaded:Cascaded human-object interaction recognition[[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhou_Cascaded_Human-Object_Interaction_Recognition_CVPR_2020_paper.html)
- [30] DRG:DRG: Dual Relation Graph for Human-Object Interaction Detection[[Paper]](https://link.springer.com/chapter/10.1007/978-3-030-58610-2_41)
- [69] MLCNet:Human object interaction detection via multi-level conditioned network[[Paper]](https://dl.acm.org/doi/abs/10.1145/3372278.3390671)
- [104] in-GraphNet:A graph-based interactive reasoning for human-object interaction detection[[Paper]](https://arxiv.org/abs/2007.06925)
- [108] VS-GATs:Visual-semantic graph attention networks for human-object interaction detection[[Paper]](https://ieeexplore.ieee.org/abstract/document/9739429)
- [31] CHGN:Contextual heterogeneous graph network for human-object interaction detection[[Paper]](https://link.springer.com/chapter/10.1007/978-3-030-58520-4_15)
- [60] VSGNet:Vsgnet: Spatial attention network for detecting human object interactions using graph convolutions[[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/html/Ulutan_VSGNet_Spatial_Attention_Network_for_Detecting_Human_Object_Interactions_Using_CVPR_2020_paper.html)
- [68] SAG:Spatio-attentive graphs for human-object interaction detection[[Paper]](https://arxiv.org/pdf/2012.06060v1)
- [87] PD-Net:Polysemy deciphering network for robust human–object interaction detection[[Paper]](https://link.springer.com/article/10.1007/s11263-021-01458-8)
- [110] PFNet:Detecting human—object interaction with multi-level pairwise feature network[[Paper]](https://link.springer.com/article/10.1007/s41095-020-0188-2)
- [105] SCG:Spatially conditioned graphs for detecting human-object interactions[[Paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Spatially_Conditioned_Graphs_for_Detecting_Human-Object_Interactions_ICCV_2021_paper.html)
- [106] SG2HOI:Exploiting scene graphs for human-object interaction detection[[Paper]](https://openaccess.thecvf.com/content/ICCV2021/html/He_Exploiting_Scene_Graphs_for_Human-Object_Interaction_Detection_ICCV_2021_paper.html)
- [92] AGRR:Action-guided attention mining and relation reasoning network for human-object interaction detection[[Paper]](https://www.academia.edu/download/97450489/0154.pdf)
- [96] OSGNet:Improved human-object interaction detection through on-the-fly stacked generalization[[Paper]](https://ieeexplore.ieee.org/abstract/document/9360596)
- [107] IGPN:Ipgn:Interactiveness proposal graph network for human-object interaction detection[[Paper]](https://ieeexplore.ieee.org/abstract/document/9489275)
- [93] Actor-centric:Effective actor-centric human-object interaction detection[[Paper]](https://www.sciencedirect.com/science/article/pii/S0262885622000518)
- [109] SGCN4HOI:A skeleton-aware graph convolutional network for human-object interaction detection[[Paper]](https://ieeexplore.ieee.org/abstract/document/9945149)
- [88] SDT:Distance matters in human-object interaction detection[[Paper]](https://dl.acm.org/doi/abs/10.1145/3503161.3547793)
- [97] Cross-Person Cues:Mining cross-person cues for body-part interactiveness learning in hoi detection[[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-19772-7_8)
- [90] PPDM++:Mining cross-person cues for body-part interactiveness learning in hoi detection[[Paper]](https://ieeexplore.ieee.org/abstract/document/10496247)
- [94] GFIN:Human–object interaction detection via global context and pairwise-level fusion features integration[[Paper]](https://www.sciencedirect.com/science/article/pii/S0893608023006251)
</details>
---
<details>
<summary>## One-stage methods</summary>

- [33] PPDM:Ppdm: Parallel point detection and matching for real-time human-object interaction detection[[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/html/Liao_PPDM_Parallel_Point_Detection_and_Matching_for_Real-Time_Human-Object_Interaction_CVPR_2020_paper.html)
- [34] IP-Net:Learning human-object interaction detection using interaction points[[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Learning_Human-Object_Interaction_Detection_Using_Interaction_Points_CVPR_2020_paper.html)
- [36] UnionDet:Uniondet: Union-level detector towards real-time human-object interaction detection[[Paper]](https://link.springer.com/chapter/10.1007/978-3-030-58555-6_30)
- [35] GG-Net:Glance and gaze: Inferring action-aware points for one-stage human-object interaction detection[[Paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Zhong_Glance_and_Gaze_Inferring_Action-Aware_Points_for_One-Stage_Human-Object_Interaction_CVPR_2021_paper.html)
- [111] DIRV:Dirv: Dense interaction region voting for end-to-end human-object interaction detection[[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/16217)
</details>
---

<details>
<summary>## Transformer-based methods</summary>

- [37] HOI-Trans:End-to-end human object interaction detection with hoi transformer[[Paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Zou_End-to-End_Human_Object_Interaction_Detection_With_HOI_Transformer_CVPR_2021_paper.html)
- [38] QPIC:Qpic: Query-based pairwise human-object interaction detection with image-wide contextual information[[Paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Tamura_QPIC_Query-Based_Pairwise_Human-Object_Interaction_Detection_With_Image-Wide_Contextual_Information_CVPR_2021_paper.html)
- [41] PST:Visual Relationship Detection Using Part-and-Sum Transformers with Composite Queries[[Paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Dong_Visual_Relationship_Detection_Using_Part-and-Sum_Transformers_With_Composite_Queries_ICCV_2021_paper.html)
- [39] HOTR:Hotr: End-to-end human-object interaction detection with transformers[[Paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Kim_HOTR_End-to-End_Human-Object_Interaction_Detection_With_Transformers_CVPR_2021_paper.html)
- [128] HORT:Detecting human-object relationships in videos[[Paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Ji_Detecting_Human-Object_Relationships_in_Videos_ICCV_2021_paper.html)
- [42] LV-HOI:Discovering human interactions with large-vocabulary objects via query and multi-scale detection[[Paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Discovering_Human_Interactions_With_Large-Vocabulary_Objects_via_Query_and_Multi-Scale_ICCV_2021_paper.html)
- [127] GTNet:Gtnet: Guided transformer network for detecting human-object interactions[[Paper]](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12527/125270Q/Gtnet-guided-transformer-network-for-detecting-human-object-interactions/10.1117/12.2663936.short)
- [40] AS-NET:Reformulating hoi detection as adaptive set prediction[[Paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Reformulating_HOI_Detection_As_Adaptive_Set_Prediction_CVPR_2021_paper.html)
- [17] QAHOI:Qahoi: Query-based anchors for human-object interaction detection[[Paper]](https://ieeexplore.ieee.org/abstract/document/10215534)
- [114] OCN:Detecting human-object interactions with object-guided cross-modal calibrated semantics[[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/20229)
- [130] MSTR:Mstr: Multi-scale transformer for end-to-end human-object interaction detection[[Paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_MSTR_Multi-Scale_Transformer_for_End-to-End_Human-Object_Interaction_Detection_CVPR_2022_paper.html)
- [113] UPT:Efficient two-stage detection of human-object interactions with a novel unary-pairwise transformer[[Paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Efficient_Two-Stage_Detection_of_Human-Object_Interactions_With_a_Novel_Unary-Pairwise_CVPR_2022_paper.html)
- [116] STIP:Exploring structure-aware transformer over interaction proposals for human-object interaction detection[[Paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Exploring_Structure-Aware_Transformer_Over_Interaction_Proposals_for_Human-Object_Interaction_Detection_CVPR_2022_paper.html)
- [129] THID:Learning transferable human-object interaction detector with natural language supervision[[Paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Learning_Transferable_Human-Object_Interaction_Detector_With_Natural_Language_Supervision_CVPR_2022_paper.html)
- [121] CATN:Category-aware transformer network for better human-object interaction detection[[Paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Dong_Category-Aware_Transformer_Network_for_Better_Human-Object_Interaction_Detection_CVPR_2022_paper.html)
- [122] CDN-S:Distillation using oracle queries for transformer-based human-object interaction detection[[Paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Qu_Distillation_Using_Oracle_Queries_for_Transformer-Based_Human-Object_Interaction_Detection_CVPR_2022_paper.html)
- [117] CPC:Consistency learning via decoding path augmentation for transformers in human object interaction detection,[[Paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Park_Consistency_Learning_via_Decoding_Path_Augmentation_for_Transformers_in_Human_CVPR_2022_paper.html)
- [118] PhraseHOI:Improving human-object interaction detection via phrase learning and label composition[[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/20041)
- [115] Disentangled Transformer:Human-object interaction detection via disentangled transformer[[Paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Zhou_Human-Object_Interaction_Detection_via_Disentangled_Transformer_CVPR_2022_paper.html)
- [119] SGPT:Sgpt: The secondary path guides the primary path in transformers for hoi detection[[Paper]](https://ieeexplore.ieee.org/abstract/document/10160329)
- [123] HOD:Hod: Human-object decoupling network for hoi detection[[Paper]](https://ieeexplore.ieee.org/abstract/document/10219794/)
- [133] MOA:Viplo: Vision transformer based pose-conditioned self-loop graph for human-object interaction detection[[Paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Park_ViPLO_Vision_Transformer_Based_Pose-Conditioned_Self-Loop_Graph_for_Human-Object_Interaction_CVPR_2023_paper.html)
- [131] MUREN:Relational context learning for human-object interaction detection[[Paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Kim_Relational_Context_Learning_for_Human-Object_Interaction_Detection_CVPR_2023_paper.html)
- [124] FGAHOI:Fgahoi: Fine-grained anchors for human-object interaction detection[[Paper]](https://ieeexplore.ieee.org/abstract/document/10315071)
- [126] PDN:Parallel disentangling network for human–object interaction detection[[Paper]](https://www.sciencedirect.com/science/article/pii/S0031320323007185)
- [125] RMRQ:Region mining and refined query improved hoi detection in transformer[[Paper]](https://ieeexplore.ieee.org/abstract/document/10516691/)
- [136] LOGICHOI:Neural-logic human-object interaction detection[[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/42b7c2f6d320d1fe1afa899a6319d6d7-Abstract-Conference.html)
- [135] CMST:Human-object interaction detection based on cascade multi-scale transformer[[Paper]](https://link.springer.com/article/10.1007/s10489-024-05324-1)
- [134] TED-Net:Ted-net: Dispersal attention for perceiving interaction region in indirectly-contact hoi detection[[Paper]](https://ieeexplore.ieee.org/abstract/document/10415065)
</details>
---

<details>
<summary>## Foundation models methods</summary>

LLM (large language models) has rich semantic knowledge. It acquires a lot of language knowledge through pre-training and provides text descriptions for HOI detection, which helps to understand the semantics of the interaction. GPT has strong zero-shot and few-shot learning capabilities. It can learn and reason without or with only a small amount of specific data, which is particularly important for solving the long-tail distribution problems. At the same time, GPT also has generative capabilities and can generate natural language descriptions to enhance the understanding of interactions. 

VLM (visual language models), such as BLIP-2 and CLIP, has strong cross-modal learning capabilities and can process visual and language information at the same time. This helps to combine visual features and natural language descriptions in HOI detection, which improves detection accuracy. At the same time, for open vocabulary HOI detection, the VLM model can use natural language descriptions to identify new and unseen interaction relationships, thereby expanding the detection capabilities.

The combination of LLM and VLM can provide richer interpretability and higher accuracy for HOI detection. It can not only intuitively display the interaction relationship, but also perform well in dealing with complex scenes and rare interaction relationships, and is better than zero-shot/few-shot learning, compositional learning, and weakly-supervised learning methods in dealing with the long-tail distribution problems.

- [160] GEN-VLKT:Gen-vlkt: Simplify association and enhance interaction understanding for hoi detection[[Paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Liao_GEN-VLKT_Simplify_Association_and_Enhance_Interaction_Understanding_for_HOI_Detection_CVPR_2022_paper.html)
- [163] Weakly-HOI:Weakly-supervised hoi detection via prior-guided bi-level representation learning[[Paper]](https://arxiv.org/abs/2303.01313)
- [162] HOICLIP:Hoiclip: Efficient knowledge transfer for hoi detection with vision-language models[[Paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Ning_HOICLIP_Efficient_Knowledge_Transfer_for_HOI_Detection_With_Vision-Language_Models_CVPR_2023_paper.html)
- [165] Interaction Labels Only:Weakly-supervised hoi detection from interaction labels only and language/vision-language priors[[Paper]](https://arxiv.org/abs/2303.05546)
- [161] EoID:End-to-end zero-shot hoi detection via vision and language knowledge distillation[[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/25385)
- [164] HOI with CLIP:Exploiting clip for zero-shot hoi detection requires knowledge distillation at multiple levels[[Paper]](https://openaccess.thecvf.com/content/WACV2024/html/Wan_Exploiting_CLIP_for_Zero-Shot_HOI_Detection_Requires_Knowledge_Distillation_at_WACV_2024_paper.html)
- [174] CLIP4HOI:Clip4hoi: Towards adapting clip for practical zero-shot hoi detection[[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8fd5bc08e744fe0dfe798c61d1575a22-Abstract-Conference.html)
- [177] ContextHOI:Contextual human object interaction understanding from pre-trained large language model[[Paper]](https://ieeexplore.ieee.org/abstract/document/10447511)
- [176] label-uncertain:Few-shot learning from augmented label-uncertain queries in bongard-hoi[[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/28079)
- [175] UniHOI:Detecting any human-object interaction relationship: Universal hoi detector with spatial prompt learning on foundation models[[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/02687e7b22abc64e651be8da74ec610e-Abstract-Conference.html)
- [178] CMD-SE:Exploring the potential of large foundation models for open-vocabulary hoi detection[[Paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Lei_Exploring_the_Potential_of_Large_Foundation_Models_for_Open-Vocabulary_HOI_CVPR_2024_paper.html)
- [179] KI2HOI:Towards zero-shot human-object interaction detection via vision-language integration[[Paper]](https://arxiv.org/abs/2403.07246)
</details>

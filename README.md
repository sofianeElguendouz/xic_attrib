# Explainability in Image Captioning based on the Latent Space
this repo contains the implementation of two attribution-based explanation techniques using the representation/latent space. Paper : [Explainability in Image Captioning based on the Latent Space]

### What can we do with this work : 
1. Perform representation space perturbation on the standard image captioning architecture using [Our previous work](https://github.com/sofianeElguendouz/RepSpaceExplanation4IC). 

2. Generate explanations for the captioning predictions using two attribution based methods based on the latent space:
2.1. LRP : we adapt the layer-wise relevance propagation method from [Explain and improve: LRP-inference fine-tuning for image captioning models](https://www.sciencedirect.com/science/article/pii/S1566253521001494) by operating in the latent space and for bottom-up based captioning models.
2.2. LIME : we develop a new version of LIME which is also based on the representation space perturbation rather than original input perturbation. This includes two types of perturbation according to their scope :
2.2.1. Visual features perturbation-based : concerns the perturbation of a given sub-set of the visual features representing the image.
2.2.2. Object perturbation-based : concerns the perturbation of entire objects rather than individual features (an object is in most cases represented by a set of visual features).

### Requirements
python >=3.6 pytorch =1.4.0

### Dataset
We use MSCOCO2017 and Flickr30k datasets. For the first part we follow the same configurations as in [Our previous work](https://github.com/sofianeElguendouz/RepSpaceExplanation4IC). For the second part, it is mendatory to download the testsets (the already encoded images) that could be found in this [google-drive repo](https://drive.google.com/drive/folders/14nmyQD3Zr7EPNyqXNLG4y8vgXClHq8Df?usp=sharing).

### Checkpoints
Please find the checkpoints for both models trained on coco2017 and flickr30k [here](https://drive.google.com/drive/folders/17qRd21PZvG7YqfT7S1FQmvD6dq1xZJey?usp=sharing)

### How to use this repo
For the first part of this work, we refer to our previous implementation that could be found [here](https://github.com/sofianeElguendouz/RepSpaceExplanation4IC). For the second part, there are several possible actions, please execute the file explain.py with the following arguments (/ to separate the supported argument values) --dataset flickr30k/coco2017 --mode expl/eval --expl_model lrp/lime1/lime2/lime3/lime4/lime5/lime_5obj/lime_8obj --eval_type correlation/ablation --abl_type ob/vf_X_min/max/15, X being the number of objects/visual features to ablate. lime1: corresponds to BU-LIME-1-2 in the paper, lime2: corresponds to BU-LIME-5, lime_5obj: BU-LIME-5-OBJ, lime_8obj: BU-LIME-8-OBJ.
1. Generate explanations for coco2017 testset using lrp method: explain.py --dataset coco2017 --mode expl --expl_model lrp
2. Evaluate the quality of the explanations using the correlation measure : explain.py --dataset coco2017 --mode eval --expl_model lrp --eval_type correlation
3. Evaluate the quality of the explanations using the ablation study (ablation of top 6 visual features using min saturation): explain.py --dataset coco2017 --mode eval --expl_model lrp --eval_type ablation --abl_type vf_6_min

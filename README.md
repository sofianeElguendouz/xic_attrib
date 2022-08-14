# Explainable image captioning using the representation space.
this repo ...

### What can we do with this repo : 
1. Perform representation space perturbation on the standard image captioning architecture (Encoder-Attention-Decoder) from [Explain and improve: LRP-inference fine-tuning for image captioning models](https://www.sciencedirect.com/science/article/pii/S1566253521001494).

2. Generate explanations for the captioning predictions using two attribution based methods based on the latent space:
2.1. LRP :
2.2. LIME :
2.2.1. Visual features perturbation-based :
2.2.2. Object perturbation-based :

### Requirements
python >=3.6 pytorch =1.4.0

### Dataset
We use MSCOCO2017 and Flickr30k datasets. For the first part we follow the same configurations as in [Our previous work](https://github.com/sofianeElguendouz/RepSpaceExplanation4IC). 

### To explain the model based on Gaussian perturbation of the components
We provide two perturbation levels (vision, language) with two components each (VF and CT, WE and HT). We refer to [rep_space_perturb.py] to perform explanation with 30 iterations to control randomness effect. Use the argument (--mode perturb) to perform explanation, and (--test_split test/train) to choose the data subset.

### To evaluate the explanation quality

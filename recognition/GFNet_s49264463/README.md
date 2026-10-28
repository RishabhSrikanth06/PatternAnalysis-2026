# GFNet for AD vs CN Classification (ADNI)
## Problem Overview

The problem is to classify the ADNI data set which contains images of patients brain slices  into AD or NC (Alzheimer's Disease or Normal Cognitive) , this classifier uses a GFnet (Global Filter Network)  with the goal of achieving a test accuracy of >0.8 with a baseline CNN to compare the results with.

## The Workings Of A GFnet As A AD/NC Classifier:
A global Filter Network works like a vision transformer-style classifier , that uses a simple frequency-domain filter instead of self-attention to share information across the whole image.It flows like this -> it takes each slice in its spatial form, cuts it into 16×16 patches, and turns each patch into a short list of 192 numbers (a token), giving a 14×14 grid of tokens. Inside each block, the grid for every feature is converted from the spatial domain to its frequency domain using a 2D FFT transform, where each value now describes how much of a particular wave pattern, from broad shapes to fine detail, the slice contains. These frequency values are multiplied by a learned filter that decides which patterns to keep, shrink or boost, and an inverse Fourier transform converts the result back into the spatial domain. Because every frequency covers the whole slice, each patch now carries information from every other patch after a single layer. A small per-patch network then recombines the features, and this block is repeated six times. Finally, the tokens are averaged into one vector and a linear layer outputs two scores (NC and AD), which a softmax turns into the confidence score.  
<img width="2055" height="1083" alt="gfnet_architecture" src="https://github.com/user-attachments/assets/2e782f6e-cc3c-4255-8d9b-a2329d56b71f" />


*Figure 1: GFNet as implemented. Left: the overall network. Right: one block. Own diagram, following Rao et al. (2023).*

## Feasibility Review

### 1 - User Need, Scope & Acceptance Criteria
The intended user for our product will mainly fall in the medical field ,with clinical researchers and diagnostic healthcare providers being the main focus of the intended user.The product is not a replacement for diagnostic health care providers but a tool they can use to help understand the urgency of which cases to review and diagnose .
The prototype classifies the ADNI data set which are MRI slices as AD or CN using a global filter network as the networks architecture . The prototype will report a confidence score with each prediction and ideally route low confidence cases to a human source. The prototype includes a baseline CNN to test whether GFNet’s extra complexity is justified, comparing accuracy, confidence, reliability, parameters, peak VRAM and inference time. It is not designed to grade Alzheimer’s severity or to process 3D scans.An open dilemma to address  is whether the GFnet is over confident in ambiguous cases and what threshold safely separates automated decisions from human review. Acceptance Criteria : 1) The GFNet achieves ≥ 0.80 test accuracy on AD vs CN . 2) Among test predictions that have a confidence score of 85% or greater the GFNet achieves a 90% test accuracy , 3)  peak training VRAM fits on a ColabPro GPU (≤ 15 GB), and inference takes ≤ 50 ms per image, 4) a confidence threshold exists that sends ≤ 30% of test cases to human review while catching ≥ 70% of GFNet’s errors.


### 2 - Model Choice & Course Concepts
The model chosen for this task is the Global Filter Network, a Hard difficulty architecture , compared against a baseline simple CNN . The model is a justified choice for this task because Alzheimer's appears as large-scale changes across the brain , and its global filters let every patch influence every other patch in a single layer whereas a CNN needs many stacked layers. It also uses 2D
FFT token mixing at roughly O(N log N) cost. This suits the VRAM and inference time criteria . These are the expected benefits right now and the comparison with the baseline CNN will test them.The proposed dataset is the preprocessed ADNI brain MRI collection (AD vs CN), split at the patient level so no patient appears in more than one set.Each MRI slice is cut into patches and projected into tokens and are passed through N stacked Global Filter Blocks.Each block applies a 2D FFT, multiplies the spectrum by a learnable complex-valued global filter, and applies a 2D IFFT, followed by an MLP with LayerNorm and residual connections, as in a ViT.final pooling layer and linear classifier give a softmax confidence score for the human-review threshold.


### 3 - Preliminary Feasibility Evidence
Preliminary Feasibility Evidence:An audit of the preprocessed ADNI data found 21,520 training slices from 1,076 patients and 9,002 test slices from 450 patients, with exactly 20 slices per patient. File names encode the patient ID and the provided train and test folders share no patient IDs, so the supplied split is patient-level and shows no leakage. No patient is labelled both AD and NC. The classes are close to balanced, at 48% AD and 52% NC in training and 50% AD and 50% NC in test, so accuracy is a meaningful metric here.


### 4 - Risks, Budget & Fallback
The main technical risks are patient-level data leakage, overfitting, and overconfidence. Slices from the same patient could end up in both training and test sets and inflate accuracy, so I will split by patient ID and check that no patient appears in more than one set. Transformer style models like GFnet can overfit on these smaller medical datasets so I will tackle this through data augmentation on training sets only. The GFnet may also be over confident on ambiguous scans which is the dilemma this project investigates, so I will measure accuracy within confidence bands and compare against the baseline CNN.If GFNet cannot reach 0.80 test accuracy or does not fit the resource limits, the fallback is a smaller configuration (fewer blocks or a smaller embedding dimension) with stronger regularisation. If it still falls short, I will report the result honestly and compare it with the baseline CNN, which is the simpler option.The next step is training the baseline CNN on the patient-level split


##  Data - Information And Preprocessing :
### 1 - Data Audit: 

The data originally started with the training set containing 1,076 patients with 21,520 slices totally , and the test set contained 450 patients with 9,002 slices totally, which is around 20 slices per patient with one patient having 22 slices.The provided data split was 85/15 with no leakage from patients in the training set to the test set. 

The image ( a slice ) is a greyscale image , containing one channel and the images are 256×240. 

Here is a example of a AD and NC greyscale image slice :

<img width="256" height="240" alt="218391_78" src="https://github.com/user-attachments/assets/33165666-1b88-4f99-8911-feb5989500b3" />

*IMAGE OF AD BRAIN SLICE*

<img width="256" height="240" alt="808819_88" src="https://github.com/user-attachments/assets/2dc61e74-e774-46a4-835c-c55e18bab589" />

*IMAGE OF NC BRAIN SLICE*


### 2 - Data split :
As mentioned before an audit of the file names showed that no patient ID appears in both sets and that no patient carries both labels, so the supplied split is already separated by patient and the test set can be trusted as an independent measure of performance. The classes are close to balanced (about 48% AD in training and 50% in test).

There is no validation set in the provided dataset , I need more feedback in training to make better choices . I cant figure out parts in the training such as , which epochs weights to keep , when to stop training and how strong the regularisation should be and where to set the confidence threshold that routes uncertain cases to a human reviewer
I created the validation set from the training set , it was separated from the remaining training data at the same strictness the test set  was .each patient contributes 20 near-identical slices and a slice-level split would let the model be validated on a brain it has already seen. The patient ID is read from each file name (218391_78.jpeg is slice 78 of patient 218391), and all of a patient’s slices are grouped together. The IDs in the data suggest that one person can appear under several numerically close IDs (probably repeat scans), so patients whose IDs are within 50 of each other are merged into one group and are never separated. The groups are shuffled with a fixed seed (42), and whole groups are moved into validation until about 15% of the patients are held out. This gives 915 training patients (18,300 slices) and 161 validation patients (3,220 slices) . 

There was no augmentation applied to the validation set.

This was how the validation set was split with the data- leakage prevention step:

<img width="686" height="558" alt="Screenshot 2026-10-28 at 3 26 18 PM" src="https://github.com/user-attachments/assets/adb144a0-24f5-4ddb-8b15-ab638acd336f" />

The assertion to check whether there are no shared patients between the validation and training set :

<img width="632" height="42" alt="Screenshot 2026-10-28 at 3 30 31 PM" src="https://github.com/user-attachments/assets/5f1b77df-f6fd-4c3d-a58d-de453ae4e6f6" />




### 3 -  Preprocessing and augmentation:
Every slice, in the training, validation and test sets, goes through the same three steps.

First - The slice is resized , 256x240 becomes 224x224 , the model needs one fixed size for its inputs. It slightly squashed the aspect ratio , which is applied equally to every set . 

Pixel values 0-225 become  0.0-1.0 , shape (1 , 224 , 224) , PyTorch needs tensors. The 1 is the single greyscale channel as mentioned in the data audit . 
 
Normalise :  (mean 0.5, std 0.5)	Values become -1 to 1. A pixel of 0.8 becomes (0.8 - 0.5) / 0.5 = 0.6
Centred inputs make training faster and steadier.

Slices that belong only to the training set had the following  augmentations applied to them :
Random rotation: up to ±15°
Random shift: up to 10% in each direction
Random zoom: a scale factor between 0.9 and 1.1
Brightness and contrast jitter: ±0.1

These mimic natural variation in head position, head size and scanner settings, so the model cannot simply memorise exact pixels. Horizontal flips were deliberately left out, because the left and right sides of the brain cannot be assumed to be interchangeable. The validation and test slices received no augmentation.

<img width="2055" height="791" alt="preprocessing_table" src="https://github.com/user-attachments/assets/d0fe5c57-131c-4b26-a60d-de0210861b8d" />

*Summary table of the preprocessing and augmentation applied to the dataset*









## Models:


## 1) BaseLine CNN:

<img width="1200" height="642" alt="0*NzssUXXCX9Vs1-YP" src="https://github.com/user-attachments/assets/9cd57032-6c18-4262-af9f-d012c8a2036d" />

* The inspired Architecture behind my Baseline CNN*
* 
### Introduction : 

The baseline for comparison with the GFnet is a convolutional neural network which I adapted from the MNIST practical in COMP3710 . It takes a 224x224 greyscale slice and passes it through four identical blocks (Added two more blocks for a better test accuracy to compare with the GFnet), each combining a 3×3 stride-2 convolution, batch normalisation, a ReLU and light dropout.The convolutions shrink the slice from 224×224 to 13×13 while the number of feature maps grows to 256, so that later blocks respond to larger structures. Global average pooling then collapses the feature maps into a 256-number summary of the slice, and a linear layer produces two logits, which a softmax turns into the NC and AD probabilities used as the confidence score. 

### Why is this a suitable baseline? 

I found that the structure of the CNN was the closest to the GFnet overall structure , they take the same 224×224 slice, pass it through a stack of repeated blocks, and end with global average pooling and a linear layer that outputs two logits and a softmax confidence score.The main difference is how information is shared across the slice , the CNN’s small 3×3 filters connect distant regions only after several layers, whereas GFNet’s global filter connects every patch to every other patch in one layer. Comparing them therefore tests whether global mixing, and GFNet’s extra parameters, are worth the cost for AD vs NC classification.

### Training Of The Baseline CNN:


The baseline was trained from scratch on the patient-level training split. After every epoch it was evaluated on the validation set, and the checkpoint with the best validation accuracy was kept. The test set was not used to choose the checkpoint.

| Setting | Value |
|---|---|
| Task | Binary classification, NC (0) vs AD (1), per slice |
| Input | 1 × 224 × 224 greyscale slice, normalised (mean 0.5, std 0.5) |
| Training data | 18,300 slices (915 patients) |
| Validation data | 3,220 slices (161 patients), grouped by nearby patient ID (seed 42) |
| Augmentation | Training set only: rotation ±15°, shift 10%, zoom 0.9–1.1, brightness/contrast ±0.1 |
| Loss function | Cross-entropy |
| Optimiser | Adam, learning rate 1e-3, no weight decay, no learning-rate schedule |
| Batch size | 32 (572 steps per epoch) |
| Epochs | 30 |
| Regularisation | Dropout 0.1, batch normalisation, augmentation |
| Weight initialisation | PyTorch defaults |
| Checkpoint rule | Best validation accuracy |
| Hardware | Google Colab Pro, NVIDIA T4 (15 GB) |
| Software | Python [x.x], PyTorch [x.x], torchvision [x.x] |
| Training time | 21.6 minutes |
| Peak GPU memory | 0.59 GB |
| Parameters | 961,666 |


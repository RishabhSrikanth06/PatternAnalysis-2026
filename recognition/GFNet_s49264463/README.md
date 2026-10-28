# GFNet for AD vs CN Classification (ADNI)
## Problem Overview
### The problem is to classify the ADNI data set which contains images of patients brain slices  into AD or NC (Alzheimer's Disease or Normal Cognitive) , this classifier uses a GFnet (Global Filter Network)  with the goal of achieving a test accuracy of >0.8 with a baseline CNN to compare the results with.

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
The data was preprocessed by   
1. Resizing :  256×240 becomes 224×224 , The model needs one fixed size. It slightly squashes the aspect ratio, which is applied equally to every set.
2. ToTensor:
Pixel values 0-255 become 0.0-1.0, shape (1, 224, 224), PyTorch needs tensors. The 1 is the single greyscale channel as mentioned in the data audit.
3. Normalise :  (mean 0.5, std 0.5)	Values become -1 to 1. A pixel of 0.8 becomes (0.8 - 0.5) / 0.5 = 0.6
Centred inputs make training faster and steadier.

Augmentation:
The training slices were randomly rotated (±10°), shifted (±5%), zoomed (0.95-1.05) and given small brightness and contrast changes to mimic variation in head position and scanner settings, while validation and test slices were only resized and normalised so their scores stay repeatable.




## Models:

### BASELINE CNN:
### INTRODUCTION:
The baseline CNN was adapted from the course MNIST practical ,  the initial two block architecture was too simple and the test accuracy was around 63% , so I upgraded it to a 4 block architecture which ended up achieving a 73% test accuracy which is a better baseline comparison.This shares a similar structure to the GFnets architecture which are repeated blocks followed by a global average pooling  and a final linear MLP layer.The small local filters in the CNN versus a global filter in the GFnet ,This lets me test whether whether GFnets extra complexity is actually justified for classifying AD against NC.

###The 4 block CNN architecture:

An input of 1×224×224 greyscale slice.
4 conv blocks, each Conv2d (3×3, stride 2) → BatchNorm → ReLU → Dropout (0.1).
Channels grow 64 → 128 → 256 → 256, and the size shrinks 224 → 111 → 55 → 27 → 13.
Global average pooling turns the 256 feature maps into 256 numbers.
A linear layer gives 2 logits (NC, AD). Softmax is applied later to get the confidence score.
961,666 parameters.

### How it was trained:

The baseline CNN was trained for 30 epochs on  google colab Pro, on a T4 gpu . Cross entropy loss was used and the Adam optimiser (with a learning rate of 1e-3) with a batch size of 32. Validation accuracy was checked after every epoch and the best checkpoint was kept, while the test set was used only once at the end.

### Why this design choice:

The baseline uses four stride-2 convolution blocks (each with BatchNorm, ReLU and light dropout) to shrink the slice from 224×224 to 13×13, followed by global average pooling and a single linear layer, which keeps the model small (961,666 parameters) and less prone to overfitting. It follows the course MNIST CNN so it stays simple, and it was extended from 2 to 4 blocks after the shallower version underfitted

### The results:
The Final 4-block baseline CNN reached a test accuracy of 73 % with a test loss of 62 %  , with the best validation accuracy is 0.860, at epoch 26. The model trained for 30 epochs which took 22.8 minutes on a Colab T4 GPU with a peak GPU memory use of 0l.59 GB. The relatively high test loss for this accuracy suggests the model’s predictions are not very confident, which is examined in the confidence analysis below.
 



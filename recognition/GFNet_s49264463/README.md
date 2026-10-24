# GFNet for AD vs CN Classification (ADNI)

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


##  Data and Preprocessing:
### 1 - Data Audit 
The data originally started with the training set containing 1,076 patients with 21,520 slices totally , and the test set contained 450 patients with 9,002 slices totally, which is around 20 slices per patient with one patient having 22 slices.The provided data split was 85/15 with no leakage from patients in the training set to the test set. The image ( a slice ) is a greyscale image , containing one channel and the images are 256×240.

### 2 - Data split :
The data is split by patient not by slice , as slices are too identical If slice 78 is in train and slice 79 is in validation, the model is “tested” on a brain it has already seen, and the score is inflated. The validation set was created form the training set where Within each class (AD / NC) , The patient IDs are sorted, shuffled with seed 42, and the first 15% are taken:
AD: 15% of 520 = 78 patients to validation
NC: 15% of 556 = about 83 patients to validation
Every slice of a chosen patient goes with them (20 each).

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



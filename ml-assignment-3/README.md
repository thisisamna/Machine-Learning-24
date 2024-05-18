# Image Classification Using SVM

## Description

Classifying images into one of 4 categories using SVM.

Two SVM models were trained. The first model was trained with a set from the original data. The second used an augmented dataset consisting of the original images, flipped versions and contrasted versions.

---

## Summary of Results

### Training metrics

Train accuracy and F1 score were both 1.0 for both models (augmented and unaugmented).

### Testing metrics

| Data used for training | # samples | Test accuracy | Test F1 Score |
| ---------------------- | --------- | ------------- | ------------- |
| Original dataset       | 57        | 0.8889        | 0.8963        |
| Augmented dataset      | 127       | 0.7778        | 0.8698        |

## Conclusion

In this case, SVM performed better without data augmentation.

This might be bacause SVM generally works well in the case of a small dataset with complex features, making data augmentation unnecessary.

Another reason might be that the type of augmentation applied was not the best suited for this case, and did not capture the true variations between the images/

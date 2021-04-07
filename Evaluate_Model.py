import os
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

testDataGen = ImageDataGenerator(rescale = 1.0/255)

validationGenerator = testDataGen.flow_from_directory(os.path.join("Splitted_Dataset", "Validation"),
                                                      target_size = (32,32),
                                                      batch_size = 32,
                                                      color_mode = "grayscale",
                                                      classes = [str(class_id) for class_id in range(59)],
                                                      class_mode = "categorical")

model = load_model('best_val_acc.hdf5')
loss, acc = model.evaluate(validationGenerator)

print("Best Accuracy Model:")
print('Loss on Validation Data : ', loss)
print('Accuracy on Validation Data :', '{:.4%}'.format(acc))

model = load_model('best_val_loss.hdf5')
loss, acc = model.evaluate(validationGenerator)

print("Best Loss Model:")
print('Loss on Validation Data : ', loss)
print('Accuracy on Validation Data :', '{:.4%}'.format(acc))

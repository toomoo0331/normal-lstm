from many_2_many import base_model

pred = base_model.Prediction(10, 5, 50, 50)
model = pred.create_model()
print(model.summary())
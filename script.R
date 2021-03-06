# ��������� ��������
install_keras(tensorflow = 'gpu')
library(keras)
library(tensorflow)

# ����������� �������
data <- dataset_mnist()
c(c(x_train, y_train), c(x_test, y_test)) %<-% data # ��������� �� ����������� � ������� �������

# �������� ��� �������� ���-������ ����������
image(x_train[11,,])

# �������, �� ������������ ����������
plotImage = function(im){
  image(t(apply(im, 2, rev)))
}
plotImage(x_train[11,,])

# ���������� ���������� ����� � ������� �� 0 �� 1
x_train = array_reshape(x_train, c(60000, 28, 28, 1))
x_train = x_train / 255

x_test  = array_reshape(x_test, c(10000, 28, 28, 1))
x_test  = x_test / 255

# �������� ����� � one-hot �������
y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)

# ������ �������� ������ �������� ������

denseModel = keras_model_sequential() %>%
  layer_flatten(input_shape = c(28,28,1)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")


denseModel %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

denseModel %>% fit(
  x_train, y_train,
  epochs = 10, batch_size=64
)

fit(denseModel, x_train, y_train, epochs=10, batch_size=64)

# ��������� �� �������� �����
result = evaluate(denseModel, x_test, y_test)
result

# ������ �� �������� ����� �� ���� ���������� ���

denseModel = keras_model_sequential() %>%
  layer_flatten(input_shape = c(28,28,1)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 64, activation = "relu") %>%	
  layer_dense(units = 10, activation = "softmax")

compile(denseModel, optimizer="adam", loss="categorical_crossentropy", metrics=c("accuracy"))

fit(denseModel, 
    x_train, y_train, 
    epochs=20, batch_size=64)

# ��������� ���� ������ �� �������� �����

result = evaluate(denseModel, x_test, y_test)
result

# �������� ����� ��������� ������. ������������ functional API

inputs = layer_input(shape=c(28,28,1))
z = layer_conv_2d(inputs, filters=32, kernel_size=c(3,3), activation="selu")
z = layer_max_pooling_2d(z, pool_size = c(2, 2))
z = layer_conv_2d(z, filters = 64, kernel_size = c(3, 3), activation = "relu")
z = layer_max_pooling_2d(z, pool_size = c(2, 2))
z = layer_conv_2d(z, filters = 64, kernel_size = c(3, 3), activation = "relu")
outputs = z
model = keras_model(inputs, outputs)

# ������ �������� ��� ��� ������������

z = layer_flatten(z)
z = layer_dense(z, units = 64, activation = "selu")
outputs = layer_dense(z, units = 10, activation = "softmax")
model = keras_model(inputs, outputs)

# ��������� ������

compile(model, optimizer="adam", loss="categorical_crossentropy", metrics=c("accuracy"))
fit(model, x_train, y_train, epochs=10, batch_size=64)

# ��������� �� �������� �����

results = model %>% evaluate(x_test, y_test)
results

# �������� �������� ������� ���� � �������� batch_size

compile(model, optimizer="adam", loss="categorical_crossentropy", metrics=c("accuracy"))
fit(model, x_train, y_train, epochs=20, batch_size=128)

# ��������� �� �������� �����

results = model %>% evaluate(x_test, y_test)
results

# �������� ������ ��� BatchNormalization

inputs = layer_input(shape=c(28,28,1))
z = layer_conv_2d(inputs, filters=32, kernel_size=c(3,3), activation="selu")
z = layer_max_pooling_2d(z, pool_size = c(2, 2))
z = layer_conv_2d(z, filters = 64, kernel_size = c(3, 3), activation = "relu")
z = layer_max_pooling_2d(z, pool_size = c(2, 2))
z = layer_batch_normalization(z)
z = layer_conv_2d(z, filters = 64, kernel_size = c(3, 3), activation = "relu")
outputs = z
model = keras_model(inputs, outputs)

# ������ �������� ��� ��� ������������

z = layer_flatten(z)
z = layer_dense(z, units = 64, activation = "selu")
outputs = layer_dense(z, units = 10, activation = "softmax")
model = keras_model(inputs, outputs)

# ��������� ������

compile(model, optimizer="adam", loss="categorical_crossentropy", metrics=c("accuracy"))
fit(model, x_train, y_train, epochs=10, batch_size=64)

# ��������� �� �������� �����

results = model %>% evaluate(x_test, y_test)
results

# �� ��� ������ ��������� �������� ������, ��� BatchNormalization

inputs = layer_input(shape=c(28,28,1))
z = layer_conv_2d(inputs, filters=32, kernel_size=c(3,3), activation="selu")
z = layer_max_pooling_2d(z, pool_size = c(2, 2))
z = layer_conv_2d(z, filters = 64, kernel_size = c(3, 3), activation = "relu")
z = layer_max_pooling_2d(z, pool_size = c(2, 2))
z = layer_conv_2d(z, filters = 64, kernel_size = c(3, 3), activation = "relu")
outputs = z
model = keras_model(inputs, outputs)

# ������ �������� ��� ��� ������������

z = layer_flatten(z)
z = layer_dense(z, units = 64, activation = "selu")
outputs = layer_dense(z, units = 10, activation = "softmax")
model = keras_model(inputs, outputs)

# ��������� ������ � ������������ sgd

compile(model, optimizer="sgd", loss="categorical_crossentropy", metrics=c("accuracy"))
fit(model, x_train, y_train, epochs=10, batch_size=64)

# ��������� �� �������� �����

results = model %>% evaluate(x_test, y_test)
results

# �������� ���������� adadelta

compile(model, optimizer="adadelta", loss="categorical_crossentropy", metrics=c("accuracy"))
fit(model, x_train, y_train, epochs=10, batch_size=64)

# ��������� �� �������� �����

results = model %>% evaluate(x_test, y_test)
results

net1=NeuralNet(
	layers=[('input',layer.InputLayer),('hidden',layer.DenseLayer),('output',layer.DenseLayer),input_shape=(None,1,1,784),hidden_num_units=1000,output_nonlinearity=lasagne.nonlinearities.softmax,output_num_units=10,update=nesterov_momentum,      update_learning_rate=0.0001,
        update_momentum=0.9,

        max_epochs=15,
        verbose=1)
		

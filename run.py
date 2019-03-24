from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import numpy as np
import IPython
import os
import tensorflow as tf
from l2_attack import CarliniL2
import time
import experiments as experiments
from all_CNN_c import All_CNN_C
from All_CNN_C_Attack import All_CNN_C_Attack,generate_data
from load_mnist import load_small_mnist, load_mnist,dataset_reshaped,dataset_attacked
import matplotlib.pyplot as plt

tf.random.set_random_seed(10)    
data_sets = load_small_mnist('data')    

num_classes = 10
input_side = 28
input_channels = 1
input_dim = input_side * input_side * input_channels 
weight_decay = 0.001
batch_size = 500

initial_learning_rate = 0.0001 
decay_epochs = [10000, 20000]
h1_units = 32
h2_units = 32
h3_units = 64
d1_units=200
d2_units=200
conv_patch_size = 3
keep_probs = [1.0, 1.0]


model_name="mnist_train_attack_inf"
model = All_CNN_C(
    input_side=input_side, 
    input_channels=input_channels,
    conv_patch_size=conv_patch_size,
    h1_units=h1_units, 
    h2_units=h2_units,
    h3_units=h3_units,
    d1_units=d1_units,
    d2_units=d2_units,
    weight_decay=weight_decay,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    damping=1e-2,
    decay_epochs=decay_epochs,
    mini_batch=True,
    train_dir='output', 
    log_dir='log',
    model_name=model_name)

num_steps = 2000

run_phase="all_layers"
if run_phase=="all_layers":
    model.train(num_steps=num_steps, iter_to_switch_to_batch=10000000,
                                     iter_to_switch_to_sgd=10000000)

iter_to_load = num_steps - 1
checkpoint_file=model.get_checkpoint()
#checkpoint_to_load = "%s-%s" % (checkpoint_file, iter_to_load) 
checkpoint_to_load=checkpoint_file+"KERAS_"+str(iter_to_load)
test_idx = 6558

if run_phase=='all_layers':
    known_indices_to_remove=[]
else:
    f=np.load('output/'+model_name+"numSteps"+num_steps+'_retraining-100.npz')
    known_indices_to_remove=f['indices_to_remove']

# actual_loss_diffs, predicted_loss_diffs, indices_to_remove = experiments.test_retraining(
#     model, 
#     test_idx=test_idx, 
#     iter_to_load=iter_to_load, 
#     num_to_remove=10,
#     num_steps=50, 
#     remove_type='maxinf',
#     known_indices_to_remove=known_indices_to_remove,  
#     force_refresh=True)

# filename1="maxinf_removed_data_numSteps"+str(num_steps)+"_"+run_phase+".txt"
# np.savetxt(filename1, np.c_[actual_loss_diffs,predicted_loss_diffs],fmt ='%f6')

# if run_phase=="all_layers":
#     np.savez(
#         'output/'+filename1, 
#         actual_loss_diffs=actual_loss_diffs, 
#         predicted_loss_diffs=predicted_loss_diffs, 
#         indices_to_remove=indices_to_remove
#         )

##################################################### NOW ATTACK! #############################################
tf.reset_default_graph()
tf_graph2=tf.Graph()

with tf_graph2.as_default() as g:
    with tf.Session(graph = g) as sess:
                
        model_to_be_attacked = All_CNN_C_Attack(
        input_side=input_side,num_classes=num_classes, num_channels=1,conv_patch_size=conv_patch_size,
        h1_units=h1_units, h2_units=h2_units,h3_units=h3_units,d1_units=d1_units,d2_units=d2_units,restore=checkpoint_to_load)
        pred=model_to_be_attacked.model.predict_classes(data_sets.test.x[:5].reshape(-1,28,28,1))
        print(pred)
        print(data_sets.test.labels[:5])
        exit()


        resized_data_sets =dataset_reshaped(data_sets)
        inputs_to_attack, targets_to_attack,input_indices = generate_data(resized_data_sets, samples=1,start=0)
        attack = CarliniL2(sess,model_to_be_attacked, batch_size=9, max_iterations=1000, confidence=0)

        timestart = time.time()
        adv_name='adv_attack_dataset_onModel_Trained_with'+str(iter_to_load)+'steps.npz'
        #if not os.path.exists(adv_name):
        adv = attack.attack(inputs_to_attack, targets_to_attack)
        print('saving adversarial attack dataset...')
        #np.savez(adv_name, adv=adv)
        timeend = time.time()
        print("Took",timeend-timestart,"seconds to run",len(inputs_to_attack),"samples.")

        print('loading adversarial attack dataset...')            

        #f=np.load(adv_name)
        #adv=f['adv']     
        for i in range(len(adv)):
            d=inputs_to_attack[i].reshape((28,28))
            e=adv[i].reshape((28,28))

            print("Correct Classification:", np.argmax(model_to_be_attacked.model.predict(inputs_to_attack[i:i+1])))
            print("Classification:", np.argmax(model_to_be_attacked.model.predict(adv[i:i+1])))
            print("Total distortion:", np.sum((adv[i]-inputs_to_attack[i])**2)**.5)
exit()
######################################################################################################################
attacked_data_set=dataset_attacked(adv,targets_to_attack)

attacked_predicted_loss_diffs= experiments.test_retraining(
    model, 
    test_idx=test_idx, 
    iter_to_load=iter_to_load, 
    num_to_remove=0,
    num_steps=None, 
    remove_type='attacked_indices',
    known_indices_to_remove=known_indices_to_remove,
    attacked_dataset=attacked_data_set,    
    force_refresh=False)

filename="attacked_removed_data_numSteps"+str(num_steps)+"_"+run_phase+".txt"
np.savetxt(filename, np.c_[attacked_predicted_loss_diffs],fmt ='%f6')


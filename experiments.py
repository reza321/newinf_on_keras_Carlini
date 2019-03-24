import numpy as np
import os
import time

import IPython
from scipy.stats import pearsonr



def test_retraining(model, test_idx, iter_to_load, force_refresh=False, 
                    num_to_remove=50, num_steps=1000, random_seed=17,
                    remove_type='',known_indices_to_remove=[],attacked_dataset=[]):

    np.random.seed(random_seed)

    model.load_checkpoint(iter_to_load)
    sess = model.sess

    
    y_test = model.data_sets.test.labels[test_idx]
    print('Test label: %s' % y_test)

    ## Or, attacked datapoints remove .
    if remove_type == 'attacked_indices':
        assert(attacked_dataset.train.x.shape[0]!=0)
        indices_to_remove = attacked_dataset
        predicted_loss_diffs = model.get_influence_on_test_loss(
            [test_idx], indices_to_remove,force_refresh=force_refresh,dataset_type="attacked")
        for i in range(len(predicted_loss_diffs)):
            print("predicted loss on attacked images: ",predicted_loss_diffs[i])
        return predicted_loss_diffs

    elif remove_type == 'maxinf':
        if len(known_indices_to_remove)!=0:
            print('you are considering part of trainable variables not all!')
            predicted_loss_diffs = model.get_influence_on_test_loss(
            [test_idx], known_indices_to_remove,
            force_refresh=force_refresh)  
        else:

            predicted_loss_diffs = model.get_influence_on_test_loss(
            [test_idx], 
            np.arange(len(model.data_sets.train.labels)),
            force_refresh=force_refresh)            
            indices_to_remove = np.argsort(np.abs(predicted_loss_diffs))[-num_to_remove:]
            predicted_loss_diffs = predicted_loss_diffs[indices_to_remove]

        actual_loss_diffs = np.zeros([num_to_remove])

        # Sanity check
        test_feed_dict = model.fill_feed_dict_with_one_ex(
            model.data_sets.test,  
            test_idx)    
        test_loss_val, params_val = sess.run([model.loss_no_reg, model.params], feed_dict=test_feed_dict)
        train_loss_val = sess.run(model.total_loss, feed_dict=model.all_train_feed_dict)
        # train_loss_val = model.minibatch_mean_eval([model.total_loss], model.data_sets.train)[0]

        model.retrain(num_steps=num_steps, feed_dict=model.all_train_feed_dict)
        retrained_test_loss_val = sess.run(model.loss_no_reg, feed_dict=test_feed_dict)
        retrained_train_loss_val = sess.run(model.total_loss, feed_dict=model.all_train_feed_dict)
        # retrained_train_loss_val = model.minibatch_mean_eval([model.total_loss], model.data_sets.train)[0]

        model.load_checkpoint(iter_to_load, do_checks=False)

        print('Sanity check: what happens if you train the model a bit more?')
        print('Loss on test idx with original model    : %s' % test_loss_val)
        print('Loss on test idx with retrained model   : %s' % retrained_test_loss_val)
        print('Difference in test loss after retraining     : %s' % (retrained_test_loss_val - test_loss_val))
        print('===')
        print('Total loss on training set with original model    : %s' % train_loss_val)
        print('Total loss on training with retrained model   : %s' % retrained_train_loss_val)
        print('Difference in train loss after retraining     : %s' % (retrained_train_loss_val - train_loss_val))
        
        print('These differences should be close to 0.\n')

        # Retraining experiment
        for counter, idx_to_remove in enumerate(indices_to_remove):

            print("=== #%s ===" % counter)
            print('Retraining without train_idx %s (label %s):' % (idx_to_remove, model.data_sets.train.labels[idx_to_remove]))

            train_feed_dict = model.fill_feed_dict_with_all_but_one_ex(model.data_sets.train, idx_to_remove)
            model.retrain(num_steps=num_steps, feed_dict=train_feed_dict)
            retrained_test_loss_val, retrained_params_val = sess.run([model.loss_no_reg, model.params], feed_dict=test_feed_dict)
            actual_loss_diffs[counter] = retrained_test_loss_val - test_loss_val
            for i in range(np.shape(params_val)[0]):
                params_val[i]=params_val[i].reshape(-1)
                retrained_params_val[i]=retrained_params_val[i].reshape(-1)
           
            print('Diff in params: %s' % np.linalg.norm(np.concatenate(params_val) - np.concatenate(retrained_params_val)))      
            print('Loss on test idx with original model    : %s' % test_loss_val)
            print('Loss on test idx with retrained model   : %s' % retrained_test_loss_val)
            print('Difference in loss after retraining     : %s' % actual_loss_diffs[counter])
            print('Predicted difference in loss (influence): %s' % predicted_loss_diffs[counter])

            # Restore params
            model.load_checkpoint(iter_to_load, do_checks=False)
            
            np.savez(
                'output/%s_loss_diffs' % model.model_name, 
                actual_loss_diffs=actual_loss_diffs, 
                predicted_loss_diffs=predicted_loss_diffs)
            print('Correlation is %s' % pearsonr(actual_loss_diffs, predicted_loss_diffs)[0])

        return actual_loss_diffs, predicted_loss_diffs, indices_to_remove
    else:
        raise ValueError('remove_type not well specified')        



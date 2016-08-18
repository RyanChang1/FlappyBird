import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import numpy as np
import tensorflow as tf
from collections import deque
import random
import time

####################
#Setting Parameters#
####################
Initial_Epsilon = 0
Final_Epsilon = 0
Max_replayMemory_size = 50000
Observe = 500
replayMemory = deque()
epsilon = Initial_Epsilon
action_class = 2
time_step = 0
batch_size = 32
gamma = 0.99
update_time = 100

FlappyBird = game.GameState()
def preprocess(observation):
	observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
	ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
	return np.reshape(observation,(80,80,1))


def setInitState(observation):
    currentState = np.stack((observation, observation, observation, observation), axis=2)
    return currentState

action0 = np.asarray([1,0])
print action0.shape
observation0, reward0, terminal = game.GameState().frame_step(action0)
observation0 = preprocess(observation0)
currentState = np.repeat(observation0,4,axis=2)
print currentState.shape


W_conv1 = tf.Variable(tf.truncated_normal([8,8,4,32], stddev = 0.01))
W_conv2 = tf.Variable(tf.truncated_normal([4,4,32,64], stddev = 0.01))
W_conv3 = tf.Variable(tf.truncated_normal([3,3,64,64], stddev = 0.01))
b_conv1 = tf.Variable(tf.constant(0.01, shape=[32]))
b_conv2 = tf.Variable(tf.constant(0.01, shape=[64]))
b_conv3 = tf.Variable(tf.constant(0.01, shape=[64]))
W_fc1 = tf.Variable(tf.truncated_normal([1600,512], stddev = 0.01))
W_fc2 = tf.Variable(tf.truncated_normal([512,2],  stddev = 0.01))
b_fc1 = tf.Variable(tf.constant(0.01, shape=[512]))
b_fc2 = tf.Variable(tf.constant(0.01, shape=[2]))

W_conv1_n = tf.Variable(tf.truncated_normal([8,8,4,32], stddev = 0.01))
W_conv2_n = tf.Variable(tf.truncated_normal([4,4,32,64], stddev = 0.01))
W_conv3_n = tf.Variable(tf.truncated_normal([3,3,64,64], stddev = 0.01))
b_conv1_n = tf.Variable(tf.constant(0.01, shape=[32]))
b_conv2_n = tf.Variable(tf.constant(0.01, shape=[64]))
b_conv3_n = tf.Variable(tf.constant(0.01, shape=[64]))
W_fc1_n = tf.Variable(tf.truncated_normal([1600,512], stddev = 0.01))
W_fc2_n = tf.Variable(tf.truncated_normal([512,2],  stddev = 0.01))
b_fc1_n = tf.Variable(tf.constant(0.01, shape=[512]))
b_fc2_n = tf.Variable(tf.constant(0.01, shape=[2]))
W_fc1_n_mean = tf.reduce_mean(W_fc1_n)

StateInput = tf.placeholder("float",[None,80,80,4])

#padding mode is SAME means output size = input_size / strides_size. It's nothing about filter_size
#So conv1 size = [20,20,32]
# pool1 size = [10,10,32]
#conv3_flat = 5*5*64 = 1600
conv1 = tf.nn.relu(tf.nn.conv2d(StateInput,filter = W_conv1,strides=[1,4,4,1],padding="SAME")+b_conv1)
pool1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
conv2 = tf.nn.relu(tf.nn.conv2d(pool1,W_conv2,[1,2,2,1],padding="SAME")+b_conv2)
conv3 = tf.nn.relu(tf.nn.conv2d(conv2,W_conv3,[1,1,1,1],padding="SAME")+b_conv3)
conv3_flat = tf.reshape(conv3,[-1,1600])
fc1 = tf.nn.relu(tf.matmul(conv3_flat,W_fc1)+b_fc1)
QValue = tf.nn.softmax(tf.matmul(fc1,W_fc2)+b_fc2)
QValue_mean = tf.reduce_mean(QValue)


Terminal = tf.placeholder("float",[None,2])
nextQValue = tf.placeholder("float",[None,2])
Gamma = tf.placeholder("float",[None,2])
#QValue_action = tf.reduce_sum(tf.mul(QValue,Action))
Reward = tf.placeholder("float",[None,2])
actionInput = tf.placeholder("float",[None,2])
#to separate QValue from min value. just take the max value to compute loss

Initial_QValue = tf.reduce_sum(tf.mul(QValue,actionInput),1)
Next_MaxQValue = tf.reduce_max(nextQValue,1)

Updated_MaxQValue = tf.nn.softmax(tf.add(Reward,tf.mul(Gamma,tf.mul(nextQValue,Terminal))))
Updated_MaxQValue_c = tf.reduce_sum(Updated_MaxQValue,1)


#Initial_QValue = tf.mul(QValue,actionInput)
#Loss = tf.reduce_mean(tf.square(tf.sub(Updated_MaxQValue,QValue)))
#Loss = tf.nn.softmax_cross_entropy_with_logits(Initial_QValue,Updated_MaxQValue)
#Loss = tf.nn.sigmoid_cross_entropy_with_logits(Initial_QValue,Updated_MaxQValue)
Loss = -tf.reduce_sum(Updated_MaxQValue*tf.log(QValue))
TrainStep = tf.train.RMSPropOptimizer(1e-6).minimize(Loss)
#TrainStep = tf.train.AdamOptimizer(1e-6).minimize(Loss)



Session = tf.InteractiveSession()
Session.run(tf.initialize_all_variables())

saver = tf.train.Saver()
'''
checkpoint = tf.train.get_checkpoint_state("saved_Parameters")
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(Session, checkpoint.model_checkpoint_path)
    print "Successfully loaded:", checkpoint.model_checkpoint_path
else :
    print "Could not find old network weights"
'''

while(True):
    Qvalue = QValue.eval(feed_dict={StateInput:[currentState]})
#    print "Qvalue", Qvalue
    action = np.zeros(2)
    if random.random() <= epsilon:
        action_index = random.randrange(action_class)
        action[action_index] = 1
    else :
        action_index = np.argmax(Qvalue)
        action[action_index] = 1
    #####################################
    #e-greedy policy is not applied here#
    #now epsilon is 0.Zero oppotunity to#
    #look for random action.            #
    #####################################
    nextobservation, reward, terminal = FlappyBird.frame_step(action)
    nextobservation = preprocess(nextobservation)
    nextState = np.append(currentState[:,:,1:],nextobservation,axis=2)
    replayMemory.append((currentState,reward,terminal,nextState,action))
    '''
    cv2.imshow('currenState0',currentState[:,:,0:1])
    cv2.imshow('currenState1',currentState[:,:,1:2])
    cv2.imshow('currenState2',currentState[:,:,2:3])
    cv2.imshow('currenState3',currentState[:,:,3:4])
    print reward,terminal
    cv2.waitKey(500)
    '''
    currentState = nextState
    #replayMemory is a type of deque and sub-type is tuple
    if len(replayMemory) > Max_replayMemory_size:
        replayMemory.popleft()
    if time_step > Observe:
        replayMemory_minibatch = random.sample(replayMemory,batch_size)
#        replayMemory_minibatch = [replayMemory[time_step]]
        state_batch = [data[0] for data in replayMemory_minibatch]
        reward_batch = [data[1] for data in replayMemory_minibatch]
        terminal_batch = [data[2] for data in replayMemory_minibatch]
        nextState_batch = [data[3] for data in replayMemory_minibatch]
        input_batch = [data[4] for data in replayMemory_minibatch]
#        print Argmax_QValue.eval(feed_dict = {StateInput:nextState_batch}).shape
        terminal_batch = np.invert(terminal_batch)
        terminal_batch = terminal_batch.astype(np.int32)
        gamma_batch = np.full([batch_size],gamma)
        Q_batch = QValue.eval(feed_dict={StateInput:nextState_batch})
        Q_batch_MaxIndex = np.argmax(Q_batch,axis=1)
        terminal_batch_expend = np.zeros([batch_size,action_class])
        gamma_batch_expend = np.ones([batch_size,action_class])
        reward_batch_expend = np.zeros([batch_size,action_class])
        for i in range(batch_size):
            terminal_batch_expend[i,Q_batch_MaxIndex[i]] = terminal_batch[i]
            gamma_batch_expend[i,Q_batch_MaxIndex[i]] = gamma_batch[i]
            reward_batch_expend[i,Q_batch_MaxIndex[i]] = reward_batch[i]
        Session.run(TrainStep,feed_dict={StateInput:state_batch, nextQValue:Q_batch, Reward:reward_batch_expend, Terminal:terminal_batch_expend, Gamma:gamma_batch_expend,
                                         actionInput:input_batch})
        '''
        print "terminal", terminal_batch_expend
        print "reward", reward_batch_expend
        print "reward_ini", reward_batch
        print "gamma", gamma_batch_expend
        print "Q_batch", Q_batch
        print "UM",Updated_MaxQValue.eval(feed_dict={StateInput:state_batch, nextQValue:Q_batch, Reward:reward_batch_expend, Terminal:terminal_batch_expend, Gamma:gamma_batch_expend})
        time.sleep(2)
        '''

        if time_step>100 and time_step%10 == 0:
#            print Loss.eval(feed_dict={StateInput:state_batch, nextStateInput:nextState_batch, Reward:reward_batch})
            print time_step, QValue.eval(feed_dict={StateInput:state_batch})
#            print "--------------------------------------"
#            print W_fc1_n_mean.eval()
    '''
    if time_step % update_time == 0:
        Session.run(
        [W_conv1_n.assign(W_conv1),
        W_conv2_n.assign(W_conv2),
        W_conv3_n.assign(W_conv3),
        b_conv1_n.assign(b_conv1),
        b_conv2_n.assign(b_conv2),
        b_conv3_n.assign(b_conv3),
        W_fc1_n.assign(W_fc1),
        W_fc2_n.assign(W_fc2),
        b_fc1_n.assign(b_fc1),
        b_fc2_n.assign(b_fc2)]
        )
    '''
    '''
    if time_step % 10000 ==0:
        saver.save(Session,'saved_Parameters/' + 'network' + '-dqn', global_step = time_step)
    '''
    time_step += 1


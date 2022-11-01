import DeepExpectedSarsa
import DeepSarsa
import DeepQL
import time
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("TkAgg")   

colors = ['cyan','aquamarine','blue','yellow','orange','red','lime','green','darkgreen']
labels = ["DES_N1","DS_N1","DQL_N1","DES_N3","DS_N3","DQL_N3","DES_N5","DS_N5","DQL_N5"]
if __name__ == '__main__':
    start_time = time.time()
    episodes = 200
    tests = 5
    env_being_tested = ["MountainCar-v0","Acrobot-v1","Pendulum-v1"]
    being_used = 1
    ai_list = [
        (env_being_tested[being_used],1,0.001,0.995,0.9,0.05,200,64,200),
        (env_being_tested[being_used],3,0.001,0.995,0.9,0.05,200,64,200),
        (env_being_tested[being_used],5,0.001,0.995,0.9,0.05,200,64,200)
    ]
    ai_types = [
        DeepExpectedSarsa.DES,
        DeepSarsa.DS,
        DeepQL.DQL
    ]
    ai_learncurve = []
    for j in range(len(ai_list)):
        k = 0
        for ai_type in ai_types:
            for i in range(tests):
                ai = ai_type(*ai_list[j])
                ai_learncurve.append(ai.train(episodes))
                print("Intermediary Done")
                # print(ai_learncurve)
            print("Done",j*len(ai_types)+k)
            k+=1

    _, ax = plt.subplots()

    for i in range(len(ai_list)*3):
        # print(ai_learncurve[i*tests:i*tests+tests])
        mean = np.array(ai_learncurve[i*tests:i*tests+tests]).mean(axis=0)
        # print(mean)
        std = np.array(ai_learncurve[i*tests:i*tests+tests]).std(axis=0)/np.sqrt(tests)
        ax.plot(range(0,episodes),mean, color=colors[i], label = labels[i])
        ax.fill_between(range(0,episodes),mean+std, mean-std, facecolor=colors[i], alpha=0.2)

    plt.xlabel("Epochs trained")
    plt.ylabel("Costs")
    plt.title("Training methods")
    ax.legend()
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.savefig('learning-curve-acrobot.png')
    plt.show()
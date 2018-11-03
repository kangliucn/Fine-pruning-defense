# Fine-Pruning Defense

This is the source code for the paper:  

[Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks](https://link.springer.com/chapter/10.1007/978-3-030-00470-5_13) (RAID 2018)  

[Kang Liu](https://engineering.nyu.edu/kang-liu), [Brendan Dolan-Gavitt](https://engineering.nyu.edu/faculty/brendan-dolan-gavitt) and [Siddharth Garg](https://engineering.nyu.edu/faculty/siddharth-garg).

If you find this code useful, please cite the paper:   

    @InProceedings{liu2018fine-pruning,
        author="Liu, Kang
        and Dolan-Gavitt, Brendan
        and Garg, Siddharth",
        title="Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks",
        booktitle="Research in Attacks, Intrusions, and Defenses",
        year="2018",
        pages="273--294",
    }
 


Training data/models for backdoor attacks on face/speech recognition can be found in the following link
https://drive.google.com/drive/folders/1GBhKk2UdeU5cB7guE4oI469JuQ573XO6?usp=sharing.  

Backdoor attacks on traffic sign classifiers can be found in https://github.com/Kooscii/BadNets.  


Please run "conv_output_prune.py" in "face" or "speech" folder to prune the network, and run "test.py" to test the network accuracy.  

Thanks to the helpful resource from https://github.com/jinze1994/DeepID1 and https://github.com/pannous/caffe-speech-recognition.

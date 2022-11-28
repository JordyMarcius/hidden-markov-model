# hidden-markov-model
The main goal from this project is to determine an abnormal pitch condition caused by :
- Internal actions such as gas and brake process 
- External actions such as potholes or obstacle on non flat road.

This algorithm is being tested using pitch data from real driving condition and can be seen in a line plot graphic below.

![7](https://user-images.githubusercontent.com/65435469/204349454-ca114e96-146f-405e-a705-a54deef1e953.PNG)

Hidden Markov Model (HMM) parameter is generated by training process in Baum-Welch algorithm. Training process will stop until the differences between value of parameter now and before is less than 0.0001. 

![2](https://user-images.githubusercontent.com/65435469/204351138-2a4cfdbe-016f-468e-85f1-cef09d1f34bf.PNG)
![3](https://user-images.githubusercontent.com/65435469/204351146-224b316a-4c9e-420e-b1d5-5c79f5de5913.PNG)
![4](https://user-images.githubusercontent.com/65435469/204351154-aa21b8c5-0abd-449e-8ed5-df72719c1ddb.PNG)

If the HMM parameter already converges, then it can be used in Viterbi algorithm to determine an abnormal pitch condition. The yellow area is an area identified as an abnormal pitch condition.

![5](https://user-images.githubusercontent.com/65435469/204351430-d751beee-934f-4868-a835-2bbbd30f664f.PNG)
![6](https://user-images.githubusercontent.com/65435469/204351736-14810f1d-5a1b-41ac-b86d-5ecece5dce23.PNG)

<h1> References </h1>
The yellow area is an area identified as an abnormal pitch condition.
L. R. Rabiner, "A tutorial on hidden Markov models and selected applications in speech recognition," in Proceedings of the IEEE, vol. 77, no. 2, pp. 257-286, Feb. 1989, doi: 10.1109/5.18626.

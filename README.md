# SAINT_plus-Knowledge-Tracing-
[SAINT+ Paper](https://arxiv.org/abs/2010.12042) Implementaions for [Riiid! Answer Correctness Prediction Competition](https://www.kaggle.com/c/riiid-test-answer-prediction) from Kaggle.


#  SAINT+: Integrating Temporal Features for EdNet Correctness Prediction
This paper added an additional features to existing architecture, SAINT: Separated Self-Attentive Neural Knowledge Tracing.</br> 
 SAINT: *Separated Self-Attentive Neural Knowledge Tracing* is from this paper [Towards an Appropriate Query, Key, and Value Computation for Knowledge Tracing](https://arxiv.org/abs/2002.07033), its pytorch implementation is [here](https://github.com/Shivanandmn/Knowledge-Tracing-SAINT) for the same dataset given above.  
#  Architecture of SAINT+
![alt text](https://github.com/Shivanandmn/SAINT_plus-Knowledge-Tracing-/blob/main/images/model.PNG?raw=true)
![alt text](https://github.com/Shivanandmn/SAINT_plus-Knowledge-Tracing-/blob/main/images/response_time.PNG?raw=true)

## To Run:
      - Change config.py- Datafile location, device, batch size. 
      - Saint.py file code is used while training in train.py directly. 
      - finally, run the train.py file in command line. 
# Citations


```bibtex
@misc{shin2020saint,
      title={SAINT+: Integrating Temporal Features for EdNet Correctness Prediction}, 
      author={Dongmin Shin and Yugeun Shim and Hangyeol Yu and Seewoo Lee and Byungsoo Kim and Youngduck Choi},
      year={2020},
      eprint={2010.12042},
      archivePrefix={arXiv},
      primaryClass={cs.CY}
}
@misc{choi2020appropriate,
      title={Towards an Appropriate Query, Key, and Value Computation for Knowledge Tracing}, 
      author={Youngduck Choi and Youngnam Lee and Junghyun Cho and Jineon Baek and Byungsoo Kim and Yeongmin Cha and Dongmin Shin and Chan Bae and Jaewe Heo},
      year={2020},
      eprint={2002.07033},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

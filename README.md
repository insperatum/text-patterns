# text-patterns

Train with
`anaconda-project run dream`

Test with
`anaconda-project run demo`

~~~~
Please enter examples:
> popopopo

model: Namespace(batch_size=500, cell_type='LSTM', embedding_size=128, hidden_size=512, max_examples=4, max_length=10, min_examples=4, mode='synthesis', name='regex_dream_anaconda-project_4826912795') iteration: 36600 score: -2.1960015296936035
-21.217   (po)+                    popo                      po                        popo                     
-21.910   (po)*                                              popopopopopo                                       
-27.902   p(op)+o                  popo                      popopo                    popo                     
-28.595   ((p)o)+                  popopo                    po                        popopopopo               
-28.595   ((po))+                  po                        popo                      popo                     
-28.595   (p(o))+                  popopo                    po                        po                       
-29.288   (po)+p?                  popo                      popop                     po                       
-29.288   (po)+2*                  po222                     po                        po22                     
-29.288   (po)+u*                  po                        po                        popou                    
-29.288   (po)+#?                  popopo                    po#                       po#        
~~~~
Todo:
- Check Robustfill should attend during P->FC rather than during softmax->P?
- give n_examples as input to FC
- allow length 0 inputs/outputs

Inference Ideas:
- Given a valid inferred regex, parse examples with that regex, choose a matching subgroup, and run inference on that parse
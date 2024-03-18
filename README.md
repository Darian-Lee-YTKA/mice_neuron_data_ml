# mice_neuron_data_ml
# Section 2 Exploratory analysis:

## Graphing average neurons across trial conditions:

After organizing the data in custom classes in python, I started my exploratory data analysis by plotting the element wise averages of the neuron matrices for each of the different experimental conditions for each session<sup>1</sup>. I defined the conditions as "left true", "right true", "left false", "right false", and "equal true", where the first part of the name corresponds to whether the left or right constrat was higher, and the second part corresponds to whether the mouse predicted it correctly. There were 5*18 graphs in total to reflect the 5 experimental conditions and the 18 sessions. 

<img src="averages_graph.png" alt="Averages Graph" width="300"/>

In this graph, I noticed that there were not many visiable differences between the trial conditions for each session. There also appeared to be some noisy neurons that were always active, such as the two bright lines in the graph above. 

<sup>1</sup> _Because individual neurons measured differ between sessions, these comparisons had to be done separately for each session_

## pairwise Welch's t-test to determine significant brain areas 

I thought these noisy neurons observed in the previous step may hint at certain brain areas that are always active and are not good predictors of the visual stimuli. Because of this, I decided to do a significance test to decide which brain areas to remove. I started by doing some research about the brain areas in order to get a better idea of what areas were likely to be important in order to ensure my finds were matching with domain intuition. For full descriptions, see my code

<img src="first_4_descriptions.png" alt="Averages Graph" width="300"/>

I took the average of each brain 

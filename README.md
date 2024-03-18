# mice_neuron_data_ml
# Section 2 Exploratory analysis:


## Graphing average neurons across trial conditions:

After organizing the data in custom classes in python, I started my exploratory data analysis by plotting the element wise averages of the neuron matrices for each of the different experimental conditions for each session<sup>1</sup>. I defined the conditions as "left true", "right true", "left false", "right false", and "equal true", where the first part of the name corresponds to whether the left or right constrat was higher, and the second part corresponds to whether the mouse predicted it correctly. There were 5*18 graphs in total to reflect the 5 experimental conditions and the 18 sessions. 

<img src="averages_graph.png" alt="Averages Graph" width="300"/>

In this graph, I noticed that there were not many visiable differences between the trial conditions for each session. There also appeared to be some noisy neurons that were always active, such as the two bright lines in the graph above. 

<sup><sup>1</sup> _Because individual neurons measured differ between sessions, these comparisons had to be done separately for each session_</sup>

## pairwise Welch's t-test to determine significant brain areas 

I thought these noisy neurons observed in the previous step may hint at certain brain areas that are always active and are not good predictors of the visual stimuli. Because of this, I decided to do a significance test to decide which brain areas to remove. I started by doing some research about the brain areas in order to get a better idea of what areas were likely to be important in order to ensure my findings in the significance test matched domain intuition. For full descriptions, see my code.

<img src="first_4_descriptions.png" alt="Averages Graph" width="900"/>

To perform the test, I extracted all the neurons corresponding to each brain area and split them up by their trial condition. Because my project aims to determine what the mice is most likely to be preciving, trials where the mouse anwsered incorrectly could skew my results since isn't clear what the mouse was percieving it as. Thus I only used the conditions 'left true', 'right true' and 'equal true'. To account for differences in when the neurons activate, I decided to store the means of each neuron's data in the first half of the .4 seconds and the mean of each neuron's data for the second half of the .4 seconds in a seperate lists. Once I had lists for all brain areas and trial conditions, I performed pairwise welches t-tests<sup>2</sup> to determine the brain areas who's mean was significantly different across trial conditions at alpha = .01<sup>3</sup>. I then plotted the data and took the union of all the brain areas that were significant at alpha = .01 for at least one comparison:

<img src="example_t_test.png" alt="example t-test output" width="900"/>

<small>_in this figure LR represents the comparision between the 'left true' and 'right true' experimental conditions, ER denotes 'equal true' compared to 'right true', etc._</small>

<img src="ttest_plotted.png" alt="plotted t-test" width="900"/>

_This graph shows the results of the significance testing for the condition 'right true' versus 'left true' for the latter .2 seconds of the neural recording. The bars represent the brain areas and their t-score associated with how much the means of the activation rate for their associated neurons differ between the two experimental conditions. The read line denotes alpha = .05 and the blue line denotes alpha = .01. I decided to use alpha = .01_

My findings allowed me to rule out 17 brain areas who were not significant differentiating any of the trial conditions. The remaining brain areas were: CP, MOp, SUB, VISl, GPe, LP, LS, CA1, ACB, LSc, AUD, MEA, VISa, VPL, SSs, MG, VISam, DG, MS, LD, ZI, RN, PL, MB, VISpm, VISrl, ACA, TT, CA3, root, VPM, NB, TH, LGd, MOs, POST, VISp, MRN, LSr, ILA, PT, MD, PO, RT, and ORB. Many of these are consistant with my initial research on brain area function and which are likely to be used in learning and visual perception. 

<sup><sup>2</sup> _welches t-tests choosen to reflect the differing sample sizes. Student t-test assumes equal variance, which is not a safe assumption when the sample sizes greatly differ as they did between 'equal' groups and 'right' or 'left' groups. This is because all of the equal groups were marked as true, where as for 'right' and 'left' we only looked at conditions where the mouse anwsered correctly. Welch's test does not assume equal variances_</sup>

<sup><sup>3</sup> _alpha = .01 choosen to limit number of features selected to reduce RAM usage and model run time_</sup>

My original plan was to use these brain areas as the features for which to bridge neurons across sessions, however, I quickly realised that each individual brain area only present in a small subset of the data with little overlap between them; thus this goal proven impractical and I decided to try clustering. 

## Clustering 

In order to ensure my clusters were not dominated by the trial situations, I created another custom class with the attribute left_right_equal to represent a 120 (3x40) element list containing the element wise averages across time for all the instances of that neuron in the trial situation 'left true' (first 40 elements), followed by the element wise averages across time for all the instances of that neuron in the trial situation 'right true' (second 40 elements), followed by the element wise averages across time for all the instances of that neuron in the trial situation 'equal true' (first last 40 elements)
Originally I tried to find three clusters, however, one of the clusters was very small, so I limited my analysis to only 2. Although the TSNE visual plot does not show much differentiation between clusters, plotting the cluster means and standard deviations as normal curves shows noticable differences<sup>4</sup>

<img src="cluster_means.png" alt="cluster means" width="500"/>

<sup><sup>3</sup> _Normal curves used since we preformed the clustering on means, and the CLT tells us that for large sample sizes, the expected value of the mean follows an approxiemently normal distribution centered at the E(xbar), which is the cluster mean_</sup>

## Creating Data Frame

Now that I had a way of grouping neurons across sessions, I was able to create a dataframe in pandas.



# **DEMETAr: Double Encoder Method for an Equivariant Transformer Architecture"**

### _I. Simion, S. Vasilev, J. Schäfer, G. Go, T. P. Kersten_

---

This blogpost serves as an introduction to our novel implementation of equivariance for transformer architectures. While equivariant transformers do already exist, we propose a method that utilizes two encoders for the node and edge information separately (which we implement in JAX mostly from scratch). This allows for more flexibility in the inputs we provide.

This blogpost serves three purposes: 
1. Explain the ideas of equivariance in transformer networks while also explaining some of the methods used.
2. Provide an overview of some reproduction results for other methods (i.e., the Equivariant Graph Neural Network).
3. Give an overview of our method and a comparison with the aforementioned reproduction results.

---

## **Equivariance in Neural Networks**

As equivariance is prevalent in the natural sciences \[1, 2, 3, 11, 17\], it makes sense to utilize them for our neural networks, especially given the evidence suggesting that it significantly improves performance through increasing the network's generalizability \[8\]. One large area within this subfield of deep learning is learning 3D translation and rotation symmetries, where various techniques have been created such as Graph Convolutional Neural Networks \[9\] and Tensor Field Networks \[10\].

Following these works, more efficient implementations have emerged, with the first being the Equivariant Graph Neural Network (EGNN) \[5\]. Based on the GNN \[4, 15, 16\], which follows a message passing scheme, it innovates by inputting the relative squared distance between two coordinates into the edge operation and to make the output equivariant, updates the coordinates of the nodes per layer. This specific method bypasses any expensive computations/approximations relative to other, similar methods while retaining high performance levels, making it preferable compared to most other GNN architectures.

More recently, transformer architectures have been utilized within the field of equivariant models. While not typically used for these types of problems due to how they were originally developed for sequential tasks \[20, 21\], recent work has suggested their effectiveness for tackling such issues \[7, 18, 19\]. This is possible through the incorporation of domain-related inductive biases, allowing them to model geometric constraints and operations. In addition, one property of transformers is that they assume full adjacency by default, which is something that can be adjusted to better match the local connectivity of GNN approaches.

Here we expand upon this idea by introducing a dual encoder architecture, where unlike most other approaches, the node and edge information are encoded separately, which are afterwards combined to a common embedding space. This provides a novel benefit in the form of learning abstract spaces from interactions between input features from the two separate modalities before seamlessly combining them.


## **<a name="recap">Recap of Equivariance</a>**

Given a set of $T_g$ transformations on a set $X$ ($T_g: X \rightarrow X$) for an element $g \in G$, where $G$ is a group acting on $X$, a function $\varphi: X \rightarrow Y$ is equivariant to $g$ iff an equivalent transformation $S_g: Y \rightarrow Y$ exists on its output space $Y$, such that:

$$\begin{align} 
\varphi(T_g(x)) = S_g(\varphi(x)). & \qquad \qquad \text{(Equation 1)}
\end{align}$$

In other words, translating the input set $T_g(x)$ and then applying $\varphi(T_x(x))$ on it yields the same result as first running the function $y = \varphi(x)$ and then applying an equivalent translation to the output $S_g(y)$ such that Equation 1 is fulfilled and $\varphi(x+g) = \varphi(x) + g$ \[5\].


## **<a name="gnns">Equivariant Graph Neural Networks</a>**

For a given graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with nodes $v_i \in \mathcal{V}$ and edges
$=e_{ij} \in \mathcal{E}$, we can define a graph convolutional layer as the following:

$$\begin{align} 
\mathbf{m}\_{ij} = \varphi_e (\mathbf{h}\_i^l, \mathbf{h}\_j^l, a_{ij}), & \qquad \qquad \text{(Equation 2)} \\
\mathbf{m}\_{i} = \sum_{j \in \mathcal{N}\_i } \mathbf{m}\_j, & \qquad \qquad \text{(Equation 3)} \\
\mathbf{h}\_i^{l+1} = \varphi_h (\mathbf{h}\_i^l, \mathbf{m}\_i), & \qquad \qquad \text{(Equation 4)}
\end{align}$$

where $\mathbf{h}\_i^l \in \mathbb{R}^{nf}$ nf is the nf-dimensional embedding of node $v_i$ at layer $l$, $a_{ij}$ are the edge attributes, $\mathcal{N}\_i$ is the set of neighbors of node $v_i$, and $\varphi_e$ and $\varphi_h$ are the
edge and node operations respectively, typically approximated by Multilayer Perceptrons (MLPs).

In order to make this implementation equivariant, \[5\] introduced the inputting of the relative squared distances between two points and updating of the node positions at each time step, leading to the following formulae:

$$\begin{align} 
\mathbf{m}\_{ij} = \varphi_e (\mathbf{h}\_i^l, \mathbf{h}\_j^l, ||\mathbf{x}\_i^l - \mathbf{x}\_j^l||^2, a_{ij}), & \qquad \qquad \text{(Equation 5)} \\
x_i^{l+1} = x_i^l + C \sum_{j \neq i} (\mathbf{x}\_i^l - \mathbf{x}\_j^l) \varphi_x(\mathbf{m}\_{ij}) , & \qquad \qquad \text{(Equation 6)} \\
\mathbf{m}\_{i} = \sum_{j \in \mathcal{N}\_i } \mathbf{m}\_j, & \qquad \qquad \text{(Equation 7)} \\
\mathbf{h}\_i^{l+1} = \varphi_h (\mathbf{h}\_i^l, \mathbf{m}\_i). & \qquad \qquad \text{(Equation 8)}
\end{align}$$

This idea of using the distances during computation forms one of the bases of our proposed transformer architecture, as it is a simple yet effective way to impose geometric equivariance within a system.

## **<a name="architecture">Equivariant Transformer</a>**

<table align="center">
  <tr align="center">
      <td><img src="assets/DEMETAr.png" width=800></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure 1.</b> Visualization of the DEMETAr architecture.</td>
  </tr>
</table>

Our method of improving the aforementioned architecture would be to leverage the capabilites of transformers \[6\]. The key difference between them and GNNs is that the former treats the entire input as a fully-connected graph. This would typically make transformers less-suited, though many papers have been published which demonstrate their effectivity in handling these tasks \[7\]. 

As our contribution to the field, we introduce a dual encoder system (visualized in Figure 1). The first encoder contains all the node features and normalized distances of each node to the molecule's center of mass, while the other exclusively encodes the edge features (i.e., bond type) and an edge length feature. 

Formally a feature vector for a single node $n$ looks like this:

$$\begin{align} 
F_n = [f_n^{(0)}, ..., f_n^{(s)}, ||x_n-x_{COM}||]
\end{align}$$

where $s$ is the number of node features, $x_i$ is the position of node $i$ and $x_{COM}$ is the center of mass position.

To explain this approach, we first need to define the following components:

$$\begin{align} 
& K^l_e, V^l_e &: \text{the keys, values of edge features at layer } l. \\
& K^l_n, V^l_n, Q^l_n &: \text{the keys, values of node features at layer } l.
\end{align}$$

Now we can begin with the actual approach. We first use an edge encoder with $p$ transformer layers on the edge features to get complex edge features. Then, we want to obtain "edge enrichments" of the node space $Z^p_e$, i.e edge information incorporated into the node encoder space. We obtain $K^p_e$, $V^p_e$ and perform the following attention operation:

$$\begin{align} 
Z^p_e = \frac{softmax(Q^p_n K^{pT}_e + M) V^p_e}{\sqrt{d}}, \qquad \qquad \text{(Equation 9)}
\end{align}$$

where the output $Z^p_e$ is a matrix of size $n \times d$ (due to the cross-attention) which contains edge encoded information in the node space for every node and $M$ is an adjacency matrix mask of size $n \times e$ where all connections are 0's and non-connections are $-\infty$ to prohibit the attention from attending to non-connected edges. Furthermore, for all layers $< p$, only the edge queries, keys, and values are used, thus no mask is required there. Contrary to that, in the $p$-th layer, we limit the attention to only the connected nodes of each edge to calculate the edge enrichment information for every node. Lastly, the final division after softmaxing by the dimension size $\sqrt(d)$ is to normalize the output scale, a method employed by most other transfomer architectures.

Now, we need to obtain the node encodings, which is done through the following: 

$$\begin{align} 
Z^r_n = \frac{softmax(Q^r_n K^{rT}_n) V^n_r}{\sqrt{d}}, \qquad \qquad \text{(Equation 10)}
\end{align}$$

where $Z^r_n$ is the output of layer $r$, which is the node encoder's last layer. Also, similar to the previous formula, we also control the output magnitude by dividing by $\sqrt{d}$.

As we now have both the node and edge features encoded, we can simply sum these encodings to combine them together:

$$\begin{align} 
Z^0_j &= Z^p_e + Z^r_n, \qquad \qquad \text{(Equation 11)}
\end{align}$$

where $Z^0_j$ is the input for a join encoder $Z^j$. This operation can alternatively be interpreted as a residual connection in the node space, where $Z^r_n$ is the residual connection. Afterwards, we continue the computation with an $h$-layer joint encoder and get the output $Z^h_j$. One final note is that we have a [CLS] token in the $Z^0_j$ or the $Z^0_n$ input which is used for classification.

Similarly to how the equivariant GNN in \[5\] is made equivariant, we created 2 different ways of introducing equivariance for a node-centric approach. Our model predicts the difference between starting and final position. To create equivariance we follow the following 2 approaches:

$$\begin{align} 
x^{output}_i = x^{input}_i + vel_i^{input} \cdot \Phi(F_i)\\
x^{output}_i = x^{input}_i + (x^{input}_i - x^{com}) \cdot \Phi(F_i)
\end{align}$$

### **Proof of Equivariance**

$$\begin{align} 
Qx_i^{update}+g&=Qx_i^{input}+g+Qvel_i^{input}\Phi(F_i)\\
&=Q(x_i^{input}+vel_i^{input}\Phi(F_i))+g\\
&=Qx_i^{update}+g\\
\end{align}$$

$$\begin{align} 
Qx_i^{update}+g&=Qx_i^{input}+g+(Qx_i^{input}+g - (Qx^{center}+g))\Phi(F_i)\\
&=Q(x_i^{input}+(x_i^{input}-x^{center})\Phi(F_i))+g\\
&=Qx_i^{update}+g\\
\end{align}$$

Our dual encoder system is equivariant is through encoding normalized distances to the molecule's center of mass and edge lengths, ensuring that the features are invariant to translations and rotations of the molecule. In addition, the attention mechanism in our transformers uses adjacency masking to ensure that attention is only paid to connected nodes and edges, which inherently respects the graph structure and maintains the relative positional information between nodes and edges. Finally, as a unique benefit of this approach, we allow for flexibility in regards to the way we accept and process inputs, due to being able to focus either only on the nodes or also the edges.

### **Experiments**

#### N-body dataset

In this dataset, a dynamical system consisting of 5 atoms is modeled in 3D space. Each atom has a positive and negative charge, a starting position and a starting velocity. The task is to predict the position of the particles after 1000 time steps. The movement of the particles follow the rules of physics: Same charges repel and different charges attract. The task is equivariant in the sense, that translating and rotating the 5-body system on the input space is the same as rotating the output space.

#### QM9 dataset

This dataset consists of small molecules and the task is to predict a chemical property. The atoms of the molecules have 3 dimensional positions and each atom is one hot encoded to the atom type. This task is an invariant task, since the chemical property does not depend on position or rotation of the molecule.


## **<a name="architecture">Evaluating the Models</a>**

As a baseline, we compare our dual encoder transformer to varying architectures, with the first being from \[5\] as it is generally the best performing model. In addition, we also show the baseline performance reported in QM9 to show how our transformer fares with other transformer methods, specifically compared with that of \[7\] as it outperforms many other implementations in the benchmarks tasks (i.e., QM9) due to utilizing radial basis functions to expand the interatomic distances and adjusting the transformer operations to acommodate to these modified distances naturally.

For all the aforementioned methods except TorchMD-Net (due to time constraints), we evaluate and reproduce their performance on the QM9 \[12, 13\] and N-body \[14\] datasets. The former is a task which involves predicting quantum chemical properties (at DFT level) of small organic molecules and is used to evaluate the model performances on invariant tasks due to only requiring property predictions. Meanwhile, the latter is to test how well each model can handle equivariance in the data, as it involves predicting the positions of particles depending on the charges and velocities.

In addition, an ablation study is conducted to evaluate the performance of our method when parts of it are disabled, which is further detailed below.

## **<a name="reproduction">Reproduction of the Experiments</a>**

To reproduce the EGNN model \[7\], we rewrote the entire model from scratch in Jax, to make use of Jax's faster just-in-time (jit) compilation.

<table align="center">
  <tr align="center">
      <th align="left">Task</th>
      <th align="left">EGNN</th>
      <th align="left">EGNN (Ours) </th>
  </tr>
  <tr align="center">
    <td align="left"> QM9 (ε<sub>HOMO</sub>) (meV)</td>
    <td align="left">29</td>
    <td align="left"></td>
  </tr>
  <tr align="center">
    <td align="left">N-Body (Position MSE)</td>
    <td align="left">0.0071</td>
    <td align="left">0.0025</td>
  </tr>
  <tr align="left">
    <td colspan=6><b>Table 1.</b> Reproduction results comparing [5] with our Jax implementation.</td>
  </tr>
</table>

Here we can see, that our EGNN implementation outperforms the original author's implementation on the N-body dataset. Using other publicly available EGNN implementations, also achieve a similar performance as our model on our data. We argue therefore, that the increased performance, comes from the fact, that the dataset is generated slightly different to the one presented in \[5\].

## **<a name="comparison">Comparison with other Methods</a>**

Meanwhile, when comparing with other transformer implementations, we see based on the below results that our method is very comparable or even outperforms many of the top approaches that have been published recently in the past few years for both QM9 and N-body. 

<table align="center">
  <tr align="center">
      <th align="left">Metric</th>
      <th align="left">WaveScatt</th>
      <th align="left">NMP</th>
      <th align="left">SchNet</th>
      <th align="left">Cormorant</th>
      <th align="left">LieConv(T3)</th>
      <th align="left">TFN</th>
      <th align="left">SE(3)-Transformer</th>
      <th align="left">TorchMD-Net</th>
      <th align="left">DEMETAr</th>
  </tr>
  <tr align="center">
    <td align="left">ε<sub>HOMO</sub> (meV)</td>
    <td align="left">85</td>
    <td align="left">43</td>
    <td align="left">41</td>
    <td align="left">34</td>
    <td align="left">30</td>
    <td align="left">40</td>
    <td align="left">35.0</td>
    <td align="left">20.3</td>
    <td align="left"></td>
  </tr>
  <tr align="left">
    <td colspan=10><b>Table 2.</b> Comparison of results for QM9, taken from [7, 18].</td>
  </tr>
</table>


<table align="center">
  <tr align="center">
      <th align="left"></th>
      <th align="left">Linear</th>
      <th align="left">DeepSet</th>
      <th align="left">Tensor Field</th>
      <th align="left">Set Transformer</th>
      <th align="left">SE(3)-Transformer</th>
      <th align="left">DEMETAr</th>
  </tr>
  <tr align="center">
    <td align="left">MSE<sub>x</sub></td>
    <td align="left">0.0691</td>
    <td align="left">0.0639</td>
    <td align="left">0.0151</td>
    <td align="left">0.0139</td>
    <td align="left"><b>0.0076</b></td>
    <td align="left">0.050895</td>
  </tr>
  <tr align="left">
    <td colspan=9><b>Table 3.</b> Comparison of results for the N-body task, taken from [18].</td>
  </tr>
</table>

## **<a name="speed">Comparison of Speed</a>**

As our method is implemented using JAX, one advantage is that it is faster than the standard PyTorch library. To show this, we compare the forward pass times of an existing implementation in PyTorch with our implementation. The results of which can be seen in the following graph:

<!-- <table align="center">
  <tr align="center">
      <td><img src="assets/speed.png" width=800></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure 1.</b> The Markov process of diffusing noise and denoising [5].</td>
  </tr>
</table> -->

Furthermore, having the implementation be fully in JAX allows it to benefit from Just-In-Time (JIT) compilation, for example in terms of helping improve the numerical stability and optimize it for even faster runtimes.

## **<a name="ablation">Ablation studies</a>**

### **Comparison of different Equivariances on the N-body dataset.**

Here, we compare 4 different transformer architectures. The first is a standard transformer (not equivariant) that uses the positions as input to predict the final positions. Furthermore, we have 3 equivariant transformers: One that is translation equivariant and 2 that are translation and rotation equivariant, one via velocity and one via distance to the center of mass. All models were trained for 40 epochs.

<table align="center">
  <tr align="center">
      <th align="left"></th>
      <th align="left">standard Transformer</th>
      <th align="left">translation equivariant Transformer</th>
      <th align="left">translation rotation equivariant Transformer center of mass</th>
    <th align="left">translation rotation equivariant Transformer velocity</th>
  </tr>
  <tr align="center">
    <td align="left">MSE<sub>x</sub></td>
    <td align="left">1.259675</td>
    <td align="left">0.364862</td>
    <td align="left">0.313850</td>
    <td align="left">0.050895</td>
  </tr>
  <tr align="left">
    <td colspan=9><b>Table 3.</b> Comparison of different equivariances on the N-body dataset.</td>
  </tr>
</table>

The baseline performance is the standard transformer. The bad performance highlights the need for equivariance for this task. The transformer has issues generalising, possibly to rotated and translated examples within a dataset. The second model, which is translation equivariant performs better however it is outperformed by models which are translation and rotation equivariant. By introducing roto translation equivariant models, they can fully learn the rules of the dynamical system while not being restricted to struggling with learning how rotations also influence the dynamical system. It is demonstrated, that models incorporating equivariance outperform those that do not. Furthermore we show that not all equivariant approaches are equally expressive.

### **Comparison of different Transformer Architectures**

Different types of architectures for the transformer are compared on the roto-translation velocity model. The hyper-parameters (hidden dimensions, number of encoders) of the models in the table were varied so that all models have around 100k parameters.

<table align="center">
  <tr align="center">
      <th align="left"></th>
      <th align="left">Node only encoder dim 128</th>
    <th align="left">Node only 4 encoder blocks</th>
      <th align="left">Edge Cross Attention</th>
      <th align="left">Double Encoder</th>
  </tr>
  <tr align="center">
    <td align="left">MSE<sub>x</sub></td>
    <td align="left">0.050895</td>
    <td align="left">0.051638</td>
    <td align="left">0.040679</td>
    <td align="left">0.050895</td>
  </tr>
  <tr align="left">
    <td colspan=9><b>Table 3.</b> Comparison of different transformer architectures on the N-body dataset for the translation rotation equivariant Transformer using velocity.</td>
  </tr>
</table>

Both Node-only encoder approaches, perform very similar, leading to the conclusion that both are able to capture the information that lies within the nodes. The best performing model, uses an embedding layer for the edge features, a node encoder built on top of a node embedding layer, both which get put into a cross attention layer. This layer enriches the node space with edge information directly, while the double encoder appraoch uses an encoder between the edge embedding and the cross attention layer.
Another aspect that is very interesting, is to see that the Node-only encoder approach with 128 hidden dimensions performs as good as the Double encoder approach. The double encoder approach enriches the node space with information from the edge space. This suggests, that 64 hidden dimensions in our models are not enough. Further experiments with a double encoder with 128 hidden dimensions (360k parameter) prove that point by having a MSE of 0.036390. The biggest constraint in our model development are the computational ressources, because of which only limited experiments were ran with a limited set of hyperparameters. 

## **Future Work**
In this section we would like to outline out theoretical vision for a method closer to the EGNN method but a still transformer. Since we did not have time to execute it in practice, this is only a theoretical overview.

A limitation to our method is that while equvariant, it does not cover the entire space of possible roto-translations, if we were to model them (as in the NBody dataset). To tackle this issue, a true E(3)-equivariant transformer could be constructed as follows.

First, the original proposed method should be modified to work in the edge space, as opposed to the node space. This means that Equation 9 will now have node keys and values and edge queries:
$$\begin{align} 
Z^p_n = \frac{softmax(Q^p_e K^{pT}_n + M) V^p_n}{\sqrt{d}},
\end{align}$$

Then, the summation of Equation 11 will be:
$$\begin{align} 
Z^0_j &= Z^p_n + Z^r_e,
\end{align}$$ 

Thus, the output of the combined encoder will have sequence length the number of edges. This allows for the correct format of outputs that fit Equation 6. Namely, the output corresponding to the edge (i,j) of the transformer will replace $\phi(m_{ij})$ in Equation 6:
$$\begin{align} 
x_i^{new} = x_i + C \sum_{j \neq i} (\mathbf{x}\_i^l - \mathbf{x}\_j^l) \Phi(F, E)_{ij},
\end{align}$$ 
where $F$ and $E$ are the node and edge feature matrices. 
Notice that the update equation is a one step formula, as opposed to the iterative update in the EGNN forumla. That is because we leave to the transformer to figure out the complex features to allow for the immediate prediction of the update coefficients.
## **Concluding Remarks**

Our equivariant transformer model (DEMETAr) provides a novel approach to encoding both node and edge information separately within transformer models, enhancing the model's ability to handle geometric constraints and operations. As such, it is quite effective for use in tasks requiring equivariance. Our method builds upon the strengths of previous approaches such as the Equivariant Graph Neural Network (EGNN) through incorporating transformer-based attention mechanisms and domain-specific inductive biases.

The reproduction of experiments on the QM9 and N-body datasets validates the effectiveness of DEMETAr, with our results demonstrating competitive performance with existing state-of-the-art methods and even outperforming many recent implementations in both invariant and equivariant tasks. Furthermore, the implementation of DEMETAr in JAX offers considerable advantages in terms of speed and numerical stability. Our comparisons reveal that the JAX-based implementation is faster than traditional PyTorch libraries, benefiting from Just-In-Time (JIT) compilation to optimize runtime performance.

In summary, DEMETAr provides a robust framework for incorporating equivariance into transformer architectures. The dual encoder approach we introduce not only preserves geometric information but also offers flexibility in input processing, leading to improved performance across various benchmark tasks. The comprehensive evaluation and competitive results highlight its potential in use for related tasks.

## **Authors' Contributions**

- Ioana: Code implementation and debugging, running the code for results, creating the figures.
- Stefan: Code and dataloader implementation and debugging, coming up with the ideas and formulae.
- Jonas: Code implementation and debugging, proposal writing, comparing implementations and searching for ideas, running the code for results.
- Gregory: Code documentation, dependency setup, assisting with comparing implementations and searching for ideas, blogpost writing.
- Thies: Equivariance test writing, proposal writing, coordinating the group.

## Bibliography

[1] Balaban, A. T. (1985). Applications of graph theory in chemistry. In Journal of Chemical Information and Computer Sciences, 25(3), pp. 334–343. American Chemical Society (ACS). https://doi.org/10.1021/ci00047a033 

[2] Gupta, P., Goel, A., Lin, J.J., Sharma, A., Wang, D., & Zadeh, R.B. (2013). WTF: the who to follow service at Twitter. Proceedings of the 22nd international conference on World Wide Web.

[3] Miller, G. A. (1995). WordNet. In Communications of the ACM (Vol. 38, Issue 11, pp. 39–41). Association for Computing Machinery (ACM). https://doi.org/10.1145/219717.219748 

[4] Gori, M., Monfardini, G., & Scarselli, F. (2005). A new model for learning in graph domains. Proceedings. 2005 IEEE International Joint Conference on Neural Networks, 2(2), 729-734.

[5] Satorras, V. G., Hoogeboom, E., & Welling, M. (2021). E(n) Equivariant Graph Neural Networks. In Proceedings of the 38th International Conference on Machine Learning, 139. https://doi.org/10.48550/ARXIV.2102.09844

[6] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., … Polosukhin, I. (2017). Attention is All you Need. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, & R. Garnett (Eds.), Advances in Neural Information Processing Systems, 30.

[7] Thölke, P., & De Fabritiis, G. (2022). Equivariant Transformers for Neural Network based Molecular Potentials. In International Conference on Learning Representations.

[8] Bronstein, M.M., Bruna, J., Cohen, T., & Veličković, P. (2021). Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges. ArXiv, abs/2104.13478.

[9] Cohen, T. & Welling, M. (2016). Group Equivariant Convolutional Networks. Proceedings of The 33rd International Conference on Machine Learning. https://proceedings.mlr.press/v48/cohenc16.html.

[10] Thomas, N., Smidt, T.E., Kearnes, S.M., Yang, L., Li, L., Kohlhoff, K., & Riley, P.F. (2018). Tensor Field Networks: Rotation- and Translation-Equivariant Neural Networks for 3D Point Clouds. ArXiv, abs/1802.08219.

[11] Maron, H., Litany, O., Chechik, G. & Fetaya, E. (2020). On Learning Sets of Symmetric Elements. roceedings of the 37th International Conference on Machine Learning. https://proceedings.mlr.press/v119/maron20a.html.

[12] Blum, L. C., & Reymond, J.-L. (2009). 970 million druglike small molecules for virtual screening in the chemical universe database GDB-13. Journal of the American Chemical Society, 131(25), 8732–8733. https://doi.org/10.1021/ja902302h 

[13] Montavon, G., Rupp, M., Gobre, V., Vazquez-Mayagoitia, A., Hansen, K., Tkatchenko, A., Müller, K.-R., & von Lilienfeld, O. A. (2013). Machine learning of molecular electronic properties in chemical compound space. New Journal of Physics, 15(9), 095003. http://stacks.iop.org/1367-2630/15/i=9/a=095003

[14] Kipf, T., Fetaya, E., Wang, K., Welling, M. & Zemel, R. (2018). Neural Relational Inference for Interacting Systems. Proceedings of the 35th International Conference on Machine Learning. https://proceedings.mlr.press/v80/kipf18a.html.

[15] Bruna, J., Zaremba, W., Szlam, A., & Lecun, Y. (2014). Spectral networks and locally connected networks on graphs. In International Conference on Learning Representations (ICLR2014), CBLS.

[16] Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks. CoRR, abs/1609.02907. http://arxiv.org/abs/1609.02907

[17] Ramakrishnan, R., Dral, P. O., Rupp, M., & von Lilienfeld, O. A. (2014). Quantum chemistry structures and properties of 134 kilo molecules. In Scientific Data (Vol. 1, Issue 1). Springer Science and Business Media LLC. https://doi.org/10.1038/sdata.2014.22 

[18] Fuchs, F. B., Worrall, D. E., Fischer, V., & Welling, M. (2020). SE(3)-transformers: 3D roto-translation equivariant attention networks. Proceedings of the 34th International Conference on Neural Information Processing Systems.

[19] Liao, Y.-L., & Smidt, T. (2023). EQUIFORMER: Equivariant graph attention transformer for 3D atomistic graphs. Proceedings of the International Conference on Learning Representations (ICLR).

[20] Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. North American Chapter of the Association for Computational Linguistics. 

[21] Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations. Proceedings of the 34th International Conference on Neural Information Processing Systems.

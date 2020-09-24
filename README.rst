.. image:: https://raw.githubusercontent.com/giotto-ai/giotto-tda/master/doc/images/tda_logo.svg
   :width: 850

Noise-to-signal
===============
The notebook contains a pipeline to build noise detection features on psuedo-periodic signals. 

We consider the problem of predicting regime changes under noise in time series data. 
Access the full story in the `blog post
<https://towardsdatascience.com/the-shape-that-survives-the-noise-f0a2a89018c6>`_ on towards data science

Task:
-----
We consider the problem of predicting regime changes under noise in time series data. 
Access the full story in the blog `post
<https://towardsdatascience.com/the-shape-that-survives-the-noise-f0a2a89018c6>`_ on towards data science
Recover the orange signal from the blue one.

.. image:: https://miro.medium.com/max/1878/1*RtxU-EpeU_7OULHN6u9MHQ.png
   :width: 500
   
Below is the Duffing oscilator which is the circuit we used to generate the blue signal. The blue signal is obtained by varying the A parameter in the duffing oscillator. We then add noise to it. 

.. image:: https://miro.medium.com/max/581/1*S2M-M-Yov523gC9qTUR_mQ.png
   :width: 500

Features to isolate the signal:
===============================
To get an idea of which feature sets are the best to predict regime changes we build four models to perform a binary classification task. Each model is built using a different set of features: two sets of features without TDA, one using only TDA features, and one with all the combined features.

Performance of TDA features:
----------------------------
In the high noise regime TDA features yielded a significant performance boost over standard feature strategies. TDA not only outperforms the standard strategies alone, it provides a clear performance boost on top of standard strategies when the two are combined.

.. image:: https://miro.medium.com/max/1132/1*_z6KNahraO6nhzBtK2If4g.png
   :width: 500

TDA Features:
-------------
- Total number of holes: for every time window we calculate a persistence diagram. It allows us to build the Betti surface counts the number of holes present in the data as a function of epsilon and time. 
- Relevant holes feature: the relevant holes feature counts the number of holes over a given threshold size (more than 70% of the maximum value).
- Amplitude of the diagram feature: we use the diagram norm as measure of the total persistence of all the holes in the diagram.
- Mean support feature: the mean of the epsilon distances yielding non-zero Betti values in the Betti surface.
- ArgMax feature: the argmax feature is the value of epsilon for which the Betti number was highest for each time window.
- Average lifetime feature: for each dimension we take the average lifetime of a hole in the persistence diagram (=Betti surface at a fixed time).

.. image:: https://miro.medium.com/max/939/1*yfrKsJqxLKqG-qsJcMTipw.png
   :width: 500

Full pipeline:
--------------

.. image:: https://miro.medium.com/max/720/1*ikqaEipVCg3X7os2FsKl6Q.png


Feature creation:
-----------------
In order to create the TDA features, we embed our time-series into a higher dimensional space using the Takens’ embedding. Each step of the rolling window is converted into a single vector in higher-dimensional space (the dimension of which is the size of the window).

.. image:: https://miro.medium.com/max/4000/1*8JoVsvYk8w5CJRfTUCbA5Q.gif
   :width: 500






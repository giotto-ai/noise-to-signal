.. image:: https://www.giotto.ai/static/vector/logo.svg
   :width: 850

noise-to-signal
===============

We consider the problem of predicting regime changes under noise in time series data. To get an idea of which feature sets are the best to predict regime changes we build four models to perform a binary classification task. Each model is built using a different set of features: two sets of features without TDA, one using only TDA features, and one with all the combined features.

Task:
-----

Recover the orange signal from the blue one.
The blue signal is obtained by varying the A parameter in the duffing oscillator. We then add noise to it.

.. image:: https://miro.medium.com/max/1878/1*RtxU-EpeU_7OULHN6u9MHQ.png
   :width: 500
   
Features to isolate the signal
------------------------------



#VAE-Omax

A new clustering model for SOMax, an application for improvisation in MaxMSP and Python.
The current SOMax Core is using Self-Organizing Maps (SOMs). Here we replace the SOMs by Variational Auto-encoders, that have proven to be very effective on clustering problems.
The implementation is done with Tensorflow.
##Requirements :
<ul>
<li>Tensorflow 2.5.0</li>
<li>music21</li>
<li>numpy</li>
<li>pandas</li>
</ul>

##How to train

<ol>
<li>Run "python createDataset" to generate the chroma dataset from <s>the midi files in data</s> the Bach chorales. TODO : leaky integrator on/off, artificial harmonics on/off</li>
<li>Run "python train.py [--epochs EPOCHS] [--coldstart COLDSTART] [--beta BETA] [--batchsize BATCHSIZE]" to train.
</li>
<li>Run "python evaluation.py" to evaluate the latent space (work in progress)</li>
</ol>
## Deep Learning networks for segmentation.



**LICENSR MIT:**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For full licence please refer to **Unet/LICENSE**

**Short descritpion:**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Custom Unet implementations for image segmentation.

**Aim:**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Provide a coherent library enlisting multiple well known Deep Learning architecutres for image Segmentation

**Progress:**

	1. A custom UNET based approach has been included atm.

**TODOs:**

	1. Reamend the layout
	2. Package the Unet to be pip installable ... 

### Project layout:

- **Unet/**
  - **Unet/** &rarr; models, training, dataloading, deployement script (slurn based proprietary script)
  - **demos/** inferecing and visualizing demos
  - **data/** &rarr; empty dir to place data for training -- please update Unet/config.py
  - **logs/** &rarr; directory required by slurm script for logging
  - **weights/** &rarr; to store training weights
  - **deps.txt** &rarr; framwork dependencies


**Depends on :**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The project do heavily depends on:

		1. Tensorflow >= 1.8
		2. Keras >=2.2.0
		3. Opencv 3.4.0
		4. Pandas, the notorious data "tabling" package

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For extra info about the python environment setup please refer to **deps.txt**. This is the entire virtual environment that I work on to build deep learning applications.
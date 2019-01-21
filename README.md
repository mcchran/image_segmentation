Deep Learning networks for segmentation.

LICENSR MIT:
	For full licence please refer to Unet/LICENSE

Custom Unet implementations for image segmentation.

Aim:
	To provide a coherent library enlisting multiple well known Deep Learning architecutres for image Segmentation

Progress:
	A custom UNET based approach has been included atm.

Project layout:
	Unet/
		Unet/     --- models, training, dataloading, deployemen script (slurn based proprietary script)
		demos/	  --- inferecing and visualizing demos
		data/	  --- empty dir to place data for training -- please update Unet/config.py
		logs/	  --- directory required by slurm script for logging
		weights/  --- to store training weights
		deps.txt  --- framwork dependencies

		
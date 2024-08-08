conda create -n dpo-diff python=3.8
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install diffusers==0.24.0
pip install accelerate==0.24.1
pip install transformers==4.35.2
pip install matplotlib==3.7.3
pip install openai==1.3.0
pip install nltk==3.8.1
pip install gpustat==1.1.1
pip install sentence-transformers
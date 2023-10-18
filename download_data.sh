echo "Download the evaluation data..."
[ ! -d "data_hub" ] && mkdir data_hub

# https://drive.google.com/file/d/1H-wReUapuUIXnJe3lUKZoN4rLxbwT_q6/view?usp=share_link
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1H-wReUapuUIXnJe3lUKZoN4rLxbwT_q6' -O ./data_hub/ljp_data.tar.gz

tar xzf data_hub/ljp_data.tar.gz -C data_hub

echo "Download model generated results..."
[ ! -d "runs" ] && mkdir runs
# https://drive.google.com/file/d/14Zfy60udBaymsYEd8zI9z236JBS1kHvt/view?usp=share_link
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=14Zfy60udBaymsYEd8zI9z236JBS1kHvt' -O ./runs/paper_version.tar.gz

tar xzf runs/paper_version.tar.gz -C runs
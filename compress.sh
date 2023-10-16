if [ ! -d "data_hub" ]; then
  mkdir data_hub
fi

# compress eval data into tar.gz
cd data_hub
tar czvf ljp_data.tar.gz ljp

cd ../runs
tar czvf paper_version.tar.gz paper_version
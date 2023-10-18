if [ ! -d "data_hub" ]; then
  mkdir data_hub
fi

# compress eval data into tar.gz
cd data_hub
[ -d "ljp" ] && tar czvf ljp_data.tar.gz ljp || echo "Error: no ljp folder"

cd ../runs
[ -d "paper_version" ] && tar czvf paper_version.tar.gz paper_version || echo "Error: no paper_version"
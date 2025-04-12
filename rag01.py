import gensim.downloader as api
model = api.load("glove-wiki-gigaword-50")
v_banana = model["banana"]
v_grape = model["grape"]
v_aeroplane = model["aeroplane"]
sim_banana_grape = model.cosine_similarities(v_banana,[v_grape])
sim_banana_aeroplane = model.cosine_similarities(v_banana,[v_aeroplane])
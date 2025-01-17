GPU=
ifdef gpus
    GPU=--gpus=$(gpus)
endif
export GPU

doc: FORCE
	cd doc && ./build_docs.sh

docker:
	docker build -t sionna -f DOCKERFILE .

install: FORCE
	pip install .

lint:
	pylint sionna/

notebook:
	export SIONNA_NO_PREVIEW=1; \
	jupyter nbconvert --to notebook --execute --inplace $(file)

run-docker:
	docker run -p 8888:8888 --privileged=true $(GPU) --env NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility --rm -it sionna

test: FORCE
	cd test && pytest

FORCE:

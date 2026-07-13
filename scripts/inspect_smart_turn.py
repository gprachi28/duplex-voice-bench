"""Print the input/output signatures and metadata of the Smart Turn v3 ONNX
model at $SMART_TURN_MODEL_PATH. Run this once after downloading the model;
the output decides whether preprocessing lives inside the graph (raw audio
input) or outside (pre-computed mel features)."""

import os
import sys

import numpy as np
import onnxruntime as ort


def main() -> None:
    path = os.environ.get("SMART_TURN_MODEL_PATH")
    if not path:
        sys.exit("SMART_TURN_MODEL_PATH is not set")

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta()

    print(f"Model path: {path}")
    print(f"Producer: {meta.producer_name!r}")
    print(f"Domain:   {meta.domain!r}")
    print(f"Description: {meta.description!r}")
    print(f"Custom metadata: {dict(meta.custom_metadata_map)}")
    print()

    print("Inputs:")
    for i in session.get_inputs():
        print(f"  name={i.name!r}  shape={i.shape}  dtype={i.type}")

    print("Outputs:")
    for o in session.get_outputs():
        print(f"  name={o.name!r}  shape={o.shape}  dtype={o.type}")

    print()
    print("Smoke inference on zeros matching the *first* input's shape "
          "(dynamic dims -> 1):")
    inp = session.get_inputs()[0]
    shape = [1 if isinstance(d, str) or d is None else d for d in inp.shape]
    dummy = np.zeros(shape, dtype=np.float32)
    outs = session.run(None, {inp.name: dummy})
    for name, arr in zip([o.name for o in session.get_outputs()], outs):
        print(f"  {name}: shape={arr.shape}  dtype={arr.dtype}  "
              f"sample={arr.flatten()[:5].tolist()}")


if __name__ == "__main__":
    main()

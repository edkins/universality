import pickle
import transformer_lens
import numpy

safe_builtins = []

class RestrictedUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        # Only allow safe classes from builtins.
        if module == "builtins" and name in safe_builtins:
            return getattr(builtins, name)
        if (module == "transformer_lens.HookedTransformerConfig" and name == 'HookedTransformerConfig' or 
            module == 'numpy.core.multiarray' and name == 'scalar' or
            module == 'torch._utils' and name == '_rebuild_tensor_v2' or
            module == 'torch.storage' and name == '_load_from_bytes' or
            module == 'numpy' and name == 'dtype' or
            module == 'collections' and name == 'defaultdict' or
            module == 'collections' and name == 'OrderedDict'
        ):
            return super().find_class(module, name)

        # Forbid everything else.
        raise pickle.UnpicklingError("class '%s.%s' is forbidden" % (module, name))

def restricted_unpickle(data):
    return RestrictedUnpickler(data).load()

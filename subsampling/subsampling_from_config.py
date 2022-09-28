from ccsubsample.subsampling.utils import *
from ccsubsample.subsampling import kdtree_subsample


class SubsamplingFromConfig:
    def __init__(self, pytorch_geom_data, config, images):
        self.pytorch_geom_data = pytorch_geom_data
        self.config = config
        self.images = images
        self.image_indices_to_keep = None
    
    def get_subsampled_pytorch_geom_data(self):
        """
        This must be called after subsample
        :return: The subsampled pytorch geometric data
        """
        return [self.pytorch_geom_data[x] for x in self.image_indices_to_keep]
        
    def subsample(self):
        method = self.config.get("sampling_method", None)
        sampling_params = self.config.get("sampling_params")
        base_save_dir = sampling_params.get("save_results_to", "processed/samplers/")
        # check to see if the subsampling has been previously done
        loaded = load_if_exists(self.config, base_save_dir, self.images)
        if loaded:
            self.image_indices_to_keep = loaded
            return
        if method == "nns":
            self.image_indices_to_keep = self.nearest_neighbor_from_config(self.pytorch_geom_data, sampling_params)
        else:
            raise ValueError("Unrecognized sampling method {} requested".format(method))
        if sampling_params.get("save", False):
            # Make the timestamp the default if no description provided
            save_sampling_results(self.config, base_save_dir,
                                  self.images, self.image_indices_to_keep)
            
    def nearest_neighbor_from_config(self, torchg_data, sampling_params):
        image_average = sampling_params.get("image_average", False)
        fingerprints, fingerprints_to_image_index = self.extract_from_torch_data(torchg_data, image_average)
        
        # If we want the energies to be extracted via config at some point
        # if (some config component):
        #     energies = extract_energies(self.torchg_data)
        
        preprocess_params = sampling_params.get("preprocess", None)
        preprocessed_data = self.preprocess_data(fingerprints, preprocess_params)
        cutoff = sampling_params.get("cutoff", None)
        data_indices_to_keep = kdtree_subsample(preprocessed_data, cutoff)
        image_indices_to_keep = get_image_indices_to_keep(data_indices_to_keep, fingerprints_to_image_index)
        return image_indices_to_keep
    
    def extract_from_torch_data(self, torchg_data, image_average):
        if image_average:
            fingerprints, fingerprints_to_image_index = average_images(torchg_data)
        else:
            fingerprints, fingerprints_to_image_index = extract_fingerprints_with_image_indices(torchg_data)
            
        return fingerprints, fingerprints_to_image_index
    
    def preprocess_data(self, fingerprints, preprocess_params):
        if not preprocess_params:
            return scale_and_standardize_data(fingerprints)
        
        if preprocess_params.get("method", None) == "PCA":
            MAX_COMPONENT_DEFAULT = 10
            TARGET_VARIANCE_DEFAULT = 0.99
            max_components = preprocess_params.get("max_component", MAX_COMPONENT_DEFAULT)

            target_variance = preprocess_params.get("target_variance", TARGET_VARIANCE_DEFAULT)
            
            reduced = reduce_dimensions_with_pca(fingerprints, max_components, target_variance)
            return scale_and_standardize_data(reduced)
        
        # return the (scaled) original fingerprints if an invalid preprocessing method is given
        else:
            return scale_and_standardize_data(fingerprints)

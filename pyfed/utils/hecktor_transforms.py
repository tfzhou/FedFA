import random
import numpy as np
import torch

from skimage.transform import rotate


class Compose:
    def __init__(self, transforms=None):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)

        return sample


class ToTensor:
    def __init__(self, mode='train'):
        if mode not in ['train', 'test']:
            raise ValueError(f"Argument 'mode' must be 'train' or 'test'. Received {mode}")
        self.mode = mode

    def __call__(self, sample):
        if self.mode == 'train':
            img, mask = sample['input'], sample['target']
            img = np.transpose(img, axes=[3, 0, 1, 2])
            mask = np.transpose(mask, axes=[3, 0, 1, 2])
            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).float()
            sample['input'], sample['target'] = img, mask

        else:  # if self.mode == 'test'
            img = sample['input']
            img = np.transpose(img, axes=[3, 0, 1, 2])
            img = torch.from_numpy(img).float()
            sample['input'] = img

        return sample


class Mirroring:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            img, mask = sample['input'], sample['target']

            n_axes = random.randint(0, 3)
            random_axes = random.sample(range(3), n_axes)

            img = np.flip(img, axis=tuple(random_axes))
            mask = np.flip(mask, axis=tuple(random_axes))

            sample['input'], sample['target'] = img.copy(), mask.copy()

        return sample


class NormalizeIntensity:

    def __call__(self, sample):
        img = sample['input']
        img[:, :, :, 0] = self.normalize_ct(img[:, :, :, 0])
        img[:, :, :, 1] = self.normalize_pt(img[:, :, :, 1])

        sample['input'] = img
        return sample

    @staticmethod
    def normalize_ct(img, th=1024.0):
        norm_img = np.clip(img, -th, th) / th
        return norm_img

    @staticmethod
    def normalize_pt(img):
        mean = np.mean(img)
        std = np.std(img)
        return (img - mean) / (std + 1e-3)


class RandomRotation:
    def __init__(self, p=0.5, angle_range=[5, 15]):
        self.p = p
        self.angle_range = angle_range

    def __call__(self, sample):
        if random.random() < self.p:
            img, mask = sample['input'], sample['target']

            num_of_seqs = img.shape[-1]
            n_axes = random.randint(1, 3)
            random_axes = random.sample([0, 1, 2], n_axes)

            for axis in random_axes:

                angle = random.randrange(*self.angle_range)
                angle = -angle if random.random() < 0.5 else angle

                for i in range(num_of_seqs):
                    img[:, :, :, i] = RandomRotation.rotate_3d_along_axis(img[:, :, :, i], angle, axis, 1)

                mask[:, :, :, 0] = RandomRotation.rotate_3d_along_axis(mask[:, :, :, 0], angle, axis, 0)

            sample['input'], sample['target'] = img, mask
        return sample

    @staticmethod
    def rotate_3d_along_axis(img, angle, axis, order):

        if axis == 0:
            rot_img = rotate(img, angle, order=order, preserve_range=True)

        if axis == 1:
            rot_img = np.transpose(img, axes=(1, 2, 0))
            rot_img = rotate(rot_img, angle, order=order, preserve_range=True)
            rot_img = np.transpose(rot_img, axes=(2, 0, 1))

        if axis == 2:
            rot_img = np.transpose(img, axes=(2, 0, 1))
            rot_img = rotate(rot_img, angle, order=order, preserve_range=True)
            rot_img = np.transpose(rot_img, axes=(1, 2, 0))

        return rot_img


class ZeroPadding:

    def __init__(self, target_shape, mode='train'):
        self.target_shape = np.array(target_shape)  # without channel dimension
        if mode not in ['train', 'test']:
            raise ValueError(f"Argument 'mode' must be 'train' or 'test'. Received {mode}")
        self.mode = mode

    def __call__(self, sample):
        if self.mode == 'train':
            img, mask = sample['input'], sample['target']

            input_shape = np.array(img.shape[:-1])  # last (channel) dimension is ignored
            d_x, d_y, d_z = self.target_shape - input_shape
            d_x, d_y, d_z = int(d_x), int(d_y), int(d_z)

            if not all(i == 0 for i in (d_x, d_y, d_z)):
                positive = [i if i > 0 else 0 for i in (d_x, d_y, d_z)]
                negative = [i if i < 0 else None for i in (d_x, d_y, d_z)]

                # padding for positive values:
                img = np.pad(img, ((0, positive[0]), (0, positive[1]), (0, positive[2]), (0, 0)), 'constant', constant_values=(0, 0))
                mask = np.pad(mask, ((0, positive[0]), (0, positive[1]), (0, positive[2]), (0, 0)), 'constant', constant_values=(0, 0))

                # cropping for negative values:
                img = img[: negative[0], : negative[1], : negative[2], :].copy()
                mask = mask[: negative[0], : negative[1], : negative[2], :].copy()

                assert img.shape[:-1] == mask.shape[:-1], f'Shape mismatch for the image {img.shape[:-1]} and mask {mask.shape[:-1]}'

                sample['input'], sample['target'] = img, mask

            return sample

        else:  # if self.mode == 'test'
            img = sample['input']

            input_shape = np.array(img.shape[:-1])  # last (channel) dimension is ignored
            d_x, d_y, d_z = self.target_shape - input_shape
            d_x, d_y, d_z = int(d_x), int(d_y), int(d_z)

            if not all(i == 0 for i in (d_x, d_y, d_z)):
                positive = [i if i > 0 else 0 for i in (d_x, d_y, d_z)]
                negative = [i if i < 0 else None for i in (d_x, d_y, d_z)]

                # padding for positive values:
                img = np.pad(img, ((0, positive[0]), (0, positive[1]), (0, positive[2]), (0, 0)), 'constant', constant_values=(0, 0))

                # cropping for negative values:
                img = img[: negative[0], : negative[1], : negative[2], :].copy()

                sample['input'] = img

            return sample


class ExtractPatch:
    """Extracts a patch of a given size from an image (4D numpy array)."""

    def __init__(self, patch_size, p_tumor=0.5):
        self.patch_size = patch_size  # without channel dimension!
        self.p_tumor = p_tumor  # probs to extract a patch with a tumor

    def __call__(self, sample):
        img = sample['input']
        mask = sample['target']

        assert all(x <= y for x, y in zip(self.patch_size, img.shape[:-1])), \
            f"Cannot extract the patch with the shape {self.patch_size} from  " \
                f"the image with the shape {img.shape}."

        # patch_size components:
        ps_x, ps_y, ps_z = self.patch_size

        if random.random() < self.p_tumor:
            # coordinates of the tumor's center:
            xs, ys, zs, _ = np.where(mask != 0)
            tumor_center_x = np.min(xs) + (np.max(xs) - np.min(xs)) // 2
            tumor_center_y = np.min(ys) + (np.max(ys) - np.min(ys)) // 2
            tumor_center_z = np.min(zs) + (np.max(zs) - np.min(zs)) // 2

            # compute the origin of the patch:
            patch_org_x = random.randint(tumor_center_x - ps_x, tumor_center_x)
            patch_org_x = np.clip(patch_org_x, 0, img.shape[0] - ps_x)

            patch_org_y = random.randint(tumor_center_y - ps_y, tumor_center_y)
            patch_org_y = np.clip(patch_org_y, 0, img.shape[1] - ps_y)

            patch_org_z = random.randint(tumor_center_z - ps_z, tumor_center_z)
            patch_org_z = np.clip(patch_org_z, 0, img.shape[2] - ps_z)
        else:
            patch_org_x = random.randint(0, img.shape[0] - ps_x)
            patch_org_y = random.randint(0, img.shape[1] - ps_y)
            patch_org_z = random.randint(0, img.shape[2] - ps_z)

        # extract the patch:
        patch_img = img[patch_org_x: patch_org_x + ps_x,
                    patch_org_y: patch_org_y + ps_y,
                    patch_org_z: patch_org_z + ps_z,
                    :].copy()

        patch_mask = mask[patch_org_x: patch_org_x + ps_x,
                     patch_org_y: patch_org_y + ps_y,
                     patch_org_z: patch_org_z + ps_z,
                     :].copy()

        assert patch_img.shape[:-1] == self.patch_size, \
            f"Shape mismatch for the patch with the shape {patch_img.shape[:-1]}, " \
                f"whereas the required shape is {self.patch_size}."

        sample['input'] = patch_img
        sample['target'] = patch_mask

        return sample


class InverseToTensor:
    def __call__(self, sample):
        output = sample['output']

        output = torch.squeeze(output)  # squeeze the batch and channel dimensions
        output = output.numpy()

        sample['output'] = output
        return sample


class CheckOutputShape:
    def __init__(self, shape=(144, 144, 144)):
        self.shape = shape

    def __call__(self, sample):
        output = sample['output']
        assert output.shape == self.shape, \
            f'Received wrong output shape. Must be {self.shape}, but received {output.shape}.'
        return sample


class ProbsToLabels:
    def __call__(self, sample):
        output = sample['output']
        output = 0*(output == 0).astype(int) + 1*(output == 1).astype(int) + 2*(output == 2).astype(int) # get binary label
        sample['output'] = output
        return sample

class GaussianNoiseTransform:
    def __init__(self, noise_variance=(0, 0.1), p_per_sample=1, p_per_channel = 1,
                 per_channel = False):
        """
        Adds additive Gaussian Noise
        :param noise_variance: variance is uniformly sampled from that range
        :param p_per_sample:
        :param p_per_channel:
        :param per_channel: if True, each channel will get its own variance sampled from noise_variance
        :param data_key:
        CAREFUL: This transform will modify the value range of your data!
        """
        self.p_per_sample = p_per_sample
        self.noise_variance = noise_variance
        self.p_per_channel = p_per_channel
        self.per_channel = per_channel

    def __call__(self, data_dict):
        img = data_dict['input']
        if np.random.uniform() < self.p_per_sample:
            data_dict['input']  = self.augment_gaussian_noise(img, self.noise_variance, self.p_per_channel, self.per_channel)

        return data_dict

    def augment_gaussian_noise(self, data_sample, noise_variance = (0, 0.1),
                               p_per_channel = 1, per_channel= False):
        if not per_channel:
            variance = noise_variance[0] if noise_variance[0] == noise_variance[1] else \
                random.uniform(noise_variance[0], noise_variance[1])
        else:
            variance = None
        for c in range(data_sample.shape[0]):
            if np.random.uniform() < p_per_channel:
                # lol good luck reading this
                variance_here = variance if variance is not None else \
                    noise_variance[0] if noise_variance[0] == noise_variance[1] else \
                        random.uniform(noise_variance[0], noise_variance[1])
                # bug fixed: https://github.com/MIC-DKFZ/batchgenerators/issues/86
                data_sample[c] = data_sample[c] + np.random.normal(0.0, variance_here, size=data_sample[c].shape)
        return data_sample

class GammaTransform:
    def __init__(self, gamma_range=(0.5, 2), invert_image=False, per_channel=False,
                 retain_stats = False, p_per_sample=1):
        """
        Augments by changing 'gamma' of the image (same as gamma correction in photos or computer monitors
        :param gamma_range: range to sample gamma from. If one value is smaller than 1 and the other one is
        larger then half the samples will have gamma <1 and the other >1 (in the inverval that was specified).
        Tuple of float. If one value is < 1 and the other > 1 then half the images will be augmented with gamma values
        smaller than 1 and the other half with > 1
        :param invert_image: whether to invert the image before applying gamma augmentation
        :param per_channel:
        :param data_key:
        :param retain_stats: Gamma transformation will alter the mean and std of the data in the patch. If retain_stats=True,
        the data will be transformed to match the mean and standard deviation before gamma augmentation. retain_stats
        can also be callable (signature retain_stats() -> bool)
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.retain_stats = retain_stats
        self.per_channel = per_channel
        self.gamma_range = gamma_range
        self.invert_image = invert_image

    def __call__(self, data_dict):
        img = data_dict['input']
        if np.random.uniform() < self.p_per_sample:
            data_dict['input'] = self.augment_gamma(img, self.gamma_range,
                                                            self.invert_image,
                                                            per_channel=self.per_channel,
                                                            retain_stats=self.retain_stats)
        return data_dict

    def augment_gamma(self, data_sample, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-7, per_channel=False,
                      retain_stats = False):
        if invert_image:
            data_sample = - data_sample

        if not per_channel:
            retain_stats_here = retain_stats() if callable(retain_stats) else retain_stats
            if retain_stats_here:
                mn = data_sample.mean()
                sd = data_sample.std()
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            minm = data_sample.min()
            rnge = data_sample.max() - minm
            data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
            if retain_stats_here:
                data_sample = data_sample - data_sample.mean()
                data_sample = data_sample / (data_sample.std() + 1e-8) * sd
                data_sample = data_sample + mn
        else:
            for c in range(data_sample.shape[0]):
                retain_stats_here = retain_stats() if callable(retain_stats) else retain_stats
                if retain_stats_here:
                    mn = data_sample[c].mean()
                    sd = data_sample[c].std()
                if np.random.random() < 0.5 and gamma_range[0] < 1:
                    gamma = np.random.uniform(gamma_range[0], 1)
                else:
                    gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
                minm = data_sample[c].min()
                rnge = data_sample[c].max() - minm
                data_sample[c] = np.power(((data_sample[c] - minm) / float(rnge + epsilon)), gamma) * float(
                    rnge + epsilon) + minm
                if retain_stats_here:
                    data_sample[c] = data_sample[c] - data_sample[c].mean()
                    data_sample[c] = data_sample[c] / (data_sample[c].std() + 1e-8) * sd
                    data_sample[c] = data_sample[c] + mn
        if invert_image:
            data_sample = - data_sample
        return data_sample


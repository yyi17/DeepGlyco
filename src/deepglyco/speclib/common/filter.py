__all__ = ["SpectrumFilter"]

from typing import Generic, Iterable, Optional, TypeVar, Union

import numpy as np
import numpy.typing as npt

from .spec import MS2SpectrumProto

MS2SpectrumType = TypeVar("MS2SpectrumType", bound=MS2SpectrumProto)


class SpectrumFilter(Generic[MS2SpectrumType]):
    def filter_spectra(
        self,
        spectra: Iterable[MS2SpectrumType],
        min_num_fragments: Optional[int] = None,
        min_precursor_mz: Optional[float] = None,
        max_precursor_mz: Optional[float] = None,
        **kwargs,
    ) -> Iterable[MS2SpectrumType]:
        for spec in spectra:
            if not (
                self.check_filter_analyte_info(spec, **kwargs)
                and self.check_filter_precursor_mz(
                    spec, min_mz=min_precursor_mz, max_mz=max_precursor_mz
                )
                and self.check_filter_num_fragments(
                    spec, min_num_fragments=min_num_fragments, **kwargs
                )
            ):
                continue

            spec = self.filter_fragments(spec, **kwargs)
            assert not isinstance(spec, np.ndarray)

            if not self.check_filter_num_fragments(
                spec, min_num_fragments=min_num_fragments, **kwargs
            ):
                continue

            yield spec

    def check_filter_analyte_info(self, spectrum: MS2SpectrumType, **kwargs):
        return True

    def check_filter_precursor_mz(
        self,
        spectrum: MS2SpectrumType,
        min_mz: Optional[float] = None,
        max_mz: Optional[float] = None,
    ):
        mz = spectrum.precursor_mz
        if mz is None:
            return False
        return (min_mz is None or mz >= min_mz) and (max_mz is None or mz <= max_mz)

    def check_filter_num_fragments(
        self,
        spectrum: MS2SpectrumType,
        min_num_fragments: Optional[int] = None,
        **kwargs,
    ):
        return min_num_fragments is None or spectrum.num_peaks >= min_num_fragments

    def filter_fragments(
        self,
        spectrum: MS2SpectrumType,
        max_num_fragments: Optional[int] = None,
        min_fragment_mz: Optional[float] = None,
        max_fragment_mz: Optional[float] = None,
        min_relative_fragment_intensity: Optional[float] = None,
        return_index: bool = False,
        **kwargs,
    ):
        fragment_index = self.filter_fragments_by_annotations(
            spectrum, **kwargs, return_index=True
        )
        assert isinstance(fragment_index, np.ndarray)

        if min_fragment_mz is not None or max_fragment_mz is not None:
            fragment_index_1 = self.filter_fragments_by_mz(
                spectrum,
                min_mz=min_fragment_mz,
                max_mz=max_fragment_mz,
                return_index=True,
            )
            assert (
                isinstance(fragment_index_1, np.ndarray)
                and fragment_index_1.dtype == np.bool_
            )
            if fragment_index.dtype == np.bool_:
                fragment_index &= fragment_index_1
            else:
                fragment_index = np.intersect1d(
                    fragment_index, np.flatnonzero(fragment_index_1)
                )

        if max_num_fragments is not None or min_relative_fragment_intensity is not None:
            if fragment_index.dtype == np.bool_:
                fragment_index = np.flatnonzero(fragment_index)
            spectrum_1 = self.filter_fragments_by_index(spectrum, fragment_index)

            fragment_index_1 = self.filter_fragments_by_intensity(
                spectrum_1,
                relative_intensity=min_relative_fragment_intensity,
                top_n=max_num_fragments,
                return_index=True,
            )
            assert isinstance(fragment_index_1, np.ndarray)
            fragment_index = fragment_index[fragment_index_1]

        if return_index:
            return fragment_index
        else:
            return self.filter_fragments_by_index(spectrum, fragment_index)

    def filter_fragments_by_index(
        self,
        spectrum: MS2SpectrumType,
        fragment_index: Union[npt.NDArray[np.bool_], npt.NDArray[np.int_]],
        invert: bool = False,
    ) -> MS2SpectrumType:
        if invert:
            if fragment_index.dtype == np.bool_:
                fragment_index = ~fragment_index
            else:
                fragment_index = np.setdiff1d(
                    np.arange(0, len(spectrum.mz)), fragment_index
                )

        if (
            fragment_index.dtype == np.bool_
            and len(fragment_index) == len(spectrum.mz)
            and fragment_index.all()
        ):
            return spectrum

        arg_dict = {
            **dict(spectrum.analyte_info()),
            "mz": spectrum.mz[fragment_index],
            "intensity": spectrum.intensity[fragment_index],
            **{k: v[fragment_index] for k, v in spectrum.frangment_annotations()},
            **dict(spectrum.spectrum_metadata()),
        }
        return spectrum.__class__(**arg_dict) # type: ignore

    def filter_fragments_by_annotations(
        self,
        spectrum: MS2SpectrumType,
        return_index: bool = False,
        **kwargs,
    ):
        if return_index:
            return np.ones_like(spectrum.mz, dtype=np.bool_)
        else:
            return spectrum

    def filter_fragments_by_mz(
        self,
        spectrum: MS2SpectrumType,
        min_mz: Optional[float] = None,
        max_mz: Optional[float] = None,
        return_index: bool = False,
    ):
        fragment_index = np.ones_like(spectrum.mz, dtype=np.bool_)
        if min_mz is not None:
            fragment_index &= spectrum.mz >= min_mz
        if max_mz is not None:
            fragment_index &= spectrum.mz <= max_mz

        if return_index:
            return fragment_index
        else:
            return self.filter_fragments_by_index(
                spectrum, fragment_index=fragment_index
            )

    def filter_fragments_by_intensity(
        self,
        spectrum: MS2SpectrumType,
        absolute_intensity: Optional[float] = None,
        relative_intensity: Optional[float] = None,
        top_n: Optional[int] = None,
        return_index: bool = False,
    ):
        if relative_intensity is not None:
            intensity = relative_intensity * spectrum.intensity.max()
            if absolute_intensity is None:
                absolute_intensity = intensity
            else:
                absolute_intensity = max(absolute_intensity, intensity)

        if absolute_intensity is not None:
            fragment_index = spectrum.intensity >= absolute_intensity
        else:
            fragment_index = np.ones_like(spectrum.intensity, dtype=np.bool_)

        if top_n is not None:
            fragment_index = np.flatnonzero(fragment_index)
            fragment_index = np.intersect1d(
                fragment_index, (-spectrum.intensity).argsort()[:top_n]
            )
            fragment_index.sort()

        if return_index:
            return fragment_index
        else:
            return self.filter_fragments_by_index(
                spectrum, fragment_index=fragment_index
            )

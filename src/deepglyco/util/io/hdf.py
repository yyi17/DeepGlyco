"""
Extracted from alphapept.io
https://github.com/MannLabs/alphapept/blob/master/alphapept/io.py
under the Apache License 2.0
https://github.com/MannLabs/alphapept/blob/master/LICENSE

Modifications:
- Version No. as an argument
- Resolve UnicodeDecodeError
- Add options to append datasets
"""

__all__ = ["HdfFile"]


import h5py
import os
import time
import numpy as np
import pandas as pd
from typing import Any, Optional


class HdfFile:
    """
    A generic class to store and retrieve on-disk
    data with an HDF container.
    """

    @property
    def original_file_name(self):
        return self.read(attr_name="original_file_name")

    @property
    def file_name(self):
        return self.__file_name

    @property
    def directory(self):
        return os.path.dirname(self.file_name)

    @property
    def creation_time(self):
        return self.read(attr_name="creation_time")

    @property
    def last_updated(self):
        return self.read(attr_name="last_updated")

    @property
    def version(self):
        return self.read(attr_name="version")

    @property
    def expected_version(self):
        return self.__expected_version

    @property
    def is_read_only(self):
        return self.__is_read_only

    @property
    def is_overwritable(self):
        return self.__is_overwritable

    def __init__(
        self,
        file_name: str,
        is_read_only: bool = True,
        is_new_file: bool = False,
        is_overwritable: bool = False,
        version: Optional[str] = "1.0",
    ):
        """Create/open a wrapper object to access HDF data.
        Args:
            file_name (str): The file_name of the HDF file.
            is_read_only (bool): If True, the HDF file cannot be modified. Defaults to True.
            is_new_file (bool): If True, an already existing file will be completely removed. Defaults to False.
            is_overwritable (bool): If True, already existing arrays will be overwritten. If False, only new data can be appended. Defaults to False.
        """
        self.__file_name = os.path.abspath(file_name)
        self.__expected_version = version
        if is_new_file:
            is_read_only = False
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
            with h5py.File(self.file_name, "w") as hdf_file:
                current_time = time.asctime()
                hdf_file.attrs["creation_time"] = current_time
                hdf_file.attrs["original_file_name"] = self.__file_name
                hdf_file.attrs["version"] = str(version)
                hdf_file.attrs["last_updated"] = current_time
        else:
            with h5py.File(self.file_name, "r") as hdf_file:
                self.check(version=self.expected_version is not None)
        if is_overwritable:
            is_read_only = False
        self.__is_read_only = is_read_only
        self.__is_overwritable = is_overwritable

    def __eq__(self, other):
        return self.file_name == other.file_name

    def __hash__(self):
        return hash(self.file_name)

    def __str__(self):
        return f"<HDF_File {self.file_name}>"

    def __repr__(self):
        return str(self)

    def check(
        self,
        version: bool = True,
        file_name: bool = True,
    ) -> list[str]:
        """Check if the `version` or `file_name` of this HDF_File have changed.
        Args:
            version (bool): If False, do not check the version. Defaults to True.
            file_name (bool): If False, do not check the file_name. Defaults to True.
        Returns:
            list: A list of warning messages stating any issues.
        """
        warning_messages = []
        if version:
            current_version = self.expected_version
            creation_version = self.version
            if creation_version != current_version:
                warning_messages.append(
                    f"{self} was created with version "
                    f"{creation_version} instead of {current_version}."
                )
        if file_name:
            if self.file_name != self.original_file_name:
                warning_messages.append(
                    f"The file name of {self} has been changed from"
                    f"{self.original_file_name} to {self.file_name}."
                )
        return warning_messages

    def read(
        self,
        group_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        attr_name: Optional[str] = None,
        return_dataset_shape: bool = False,
        return_dataset_dtype: bool = False,
        return_dataset_slice: Any = slice(None),
        swmr: bool = False,
    ) -> Any:
        """Read contents of an HDF_File.
        Args:
            group_name (str): The group_name from where to read data.
                If no `group_name` has been provided, read directly from the root group.
                Defaults to None.
            dataset_name (str): The dataset to read.
                If no `dataset_name` has been provided, read directly from the group.
                If the `dataset_name` refers to a group, it is assumed to be
                pd.DataFrame and returned as such.
                Defaults to None.
            attr_name (str): The attribute to read.
                If `attr_name` is not None, read the attribute value instead of the contents of a group or dataset.
                If `attr_name` == "", read all attributes as a dict.
                Defaults to None.
            return_dataset_shape (bool): Do not read complete dataset to minimize RAM and IO usage.
                Defaults to False.
            return_dataset_dtype (bool): Do not read complete dataset to minimize RAM and IO usage.
                Defaults to False.
            return_dataset_slice (slice): Do not read complete dataset to minimize RAM and IO usage.
                Defaults to slice(None).
            swmr (bool): Use swmr mode to read data. Defaults to False.
        Returns:
            type: Depending on what is requested, a dict, value, np.ndarray or pd.dataframe is returned.
        Raises:
            KeyError: When the group_name does not exist.
            KeyError: When the attr_name does not exist in the group or dataset.
            KeyError: When the dataset_name does not exist in the group.
            ValueError: When the requested dataset is not a np.ndarray or pd.dataframe.
        """
        with h5py.File(self.file_name, "r", swmr=swmr) as hdf_file:
            if group_name is None:
                group: Any = hdf_file
                group_name = "/"
            else:
                try:
                    group = hdf_file[group_name]
                except KeyError as k:
                    raise KeyError(
                        f"Group {group_name} does not exist in {self}. Error {k}"
                    )
            if dataset_name is None:
                if attr_name is None:
                    return sorted(group)
                elif attr_name != "":
                    try:
                        return group.attrs[attr_name]
                    except KeyError:
                        raise KeyError(
                            f"Attribute {attr_name} does not exist for "
                            f"group {group_name} of {self}."
                        )
                else:
                    return dict(group.attrs)
            else:
                try:
                    dataset = group[dataset_name]
                except KeyError:
                    raise KeyError(
                        f"Dataset {dataset_name} does not exist for "
                        f"group {group_name} of {self}."
                    )
                if attr_name is None:
                    if isinstance(dataset, h5py.Dataset):
                        if return_dataset_shape:
                            return dataset.shape
                        elif return_dataset_dtype:
                            return dataset.dtype
                        else:
                            array = dataset[return_dataset_slice]
                            # TODO: This assumes any object array is a string array
                            if array.dtype == object:
                                # resolve UnicodeDecodeError
                                try:
                                    array = array.astype(str)
                                except UnicodeDecodeError:
                                    array = np.char.decode(array.astype(bytes))
                            return array
                    elif dataset.attrs["is_pd_dataframe"]:
                        if return_dataset_shape:
                            columns = list(dataset)
                            return (len(dataset[columns[0]]), len(columns))
                        elif return_dataset_dtype:
                            return [dataset[column].dtype for column in sorted(dataset)]
                        else:
                            df = pd.DataFrame(
                                {
                                    column: dataset[column][return_dataset_slice]
                                    for column in sorted(dataset)
                                }
                            )
                            # TODO: This assumes any object array is a string array
                            for column in dataset:
                                if df[column].dtype == object:
                                    df[column] = df[column].apply(
                                        lambda x: x
                                        if isinstance(x, str)
                                        else x.decode("UTF-8")
                                    )
                            return df
                    else:
                        raise ValueError(
                            f"{dataset_name} is not a valid dataset in "
                            f"group {group_name} of {self}."
                        )
                elif attr_name != "":
                    try:
                        return dataset.attrs[attr_name]
                    except KeyError:
                        raise KeyError(
                            f"Attribute {attr_name} does not exist for "
                            f"dataset {dataset_name} of group "
                            f"{group_name} of {self}."
                        )
                else:
                    return dict(dataset.attrs)

    def write(
        self,
        value,
        group_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        attr_name: Optional[str] = None,
        overwrite: Optional[bool] = None,
        append: bool = False,
        dataset_compression: Optional[str] = None,
        dataset_resizable: bool = False,
        swmr: bool = False,
        **dataset_kwargs,
    ) -> None:
        """Write a `value` to an HDF_File.
        Args:
            value (type): The name of the data to write.
                If the `value` is pd.DataFrame, a `dataset_name` must be provided.
            group_name (str): The group where to write data.
                If no `group_name` is provided, write directly to the root group.
                Defaults to None.
            dataset_name (str): If no `dataset_name` is provided,
                create a new group with `value` as name.
                The dataset where to write data. Defaults to None.
            attr_name (str): The attr where to write data. Defaults to None.
            overwrite (bool): Overwrite pre-existing data and truncate existing groups.
                If the False, ignore the is_overwritable flag of this HDF_File.
                Defaults to None.
            dataset_compression (str): The compression type to use for datasets.
                Defaults to None.
            swmr (bool): Open files in swmr mode. Defaults to False.
        Raises:
            IOError: When the object is read-only.
            KeyError: When the group_name or attr_name does not exist.
            ValueError: When trying to overwrite something while overwiting is disabled.
        """
        if self.is_read_only:
            raise IOError(f"Trying to write to {self}, which is read_only.")
        if overwrite is None:
            overwrite = self.is_overwritable
        if dataset_resizable:
            if "maxshape" not in dataset_kwargs:
                dataset_kwargs["maxshape"] = (None,)
        with h5py.File(self.file_name, "a", swmr=swmr) as hdf_file:

            if group_name is None:
                group: Any = hdf_file
                group_name = "/"
            else:
                try:
                    group = hdf_file[group_name]
                except KeyError:
                    raise KeyError(f"Group {group_name} does not exist in {self}.")
            if dataset_name is None:
                if attr_name is None:
                    if value in group:
                        if overwrite:
                            del group[value]
                        else:
                            raise ValueError(
                                f"New group {value} already exists in group "
                                f"{group_name} of {self}."
                            )
                    group.create_group(value)
                else:
                    if (attr_name in group.attrs) and not overwrite:
                        raise ValueError(
                            f"Attribute {attr_name} already exists in group "
                            f"{group_name} of {self}."
                        )
                    try:
                        group.attrs[attr_name] = value
                    except TypeError:
                        group.attrs[attr_name] = str(value)
            else:
                if attr_name is None:
                    if not append:
                        if dataset_name in group:
                            if overwrite:
                                del group[dataset_name]
                            else:
                                raise ValueError(
                                    f"Dataset {dataset_name} already exists in group "
                                    f"{group_name} of {self}."
                                )
                        if isinstance(value, pd.DataFrame):
                            new_group_name = f"{group_name}/{dataset_name}"
                            self.write(
                                dataset_name,
                                group_name=group_name,
                                overwrite=overwrite,
                            )
                            self.write(
                                True,
                                group_name=new_group_name,
                                attr_name="is_pd_dataframe",
                                overwrite=overwrite,
                            )
                            for column in value.columns:
                                self.write(
                                    value[column].values,
                                    group_name=new_group_name,
                                    dataset_name=column,
                                    overwrite=overwrite,
                                    dataset_compression=dataset_compression,
                                    **dataset_kwargs,
                                )
                        else:
                            dtype = value.dtype
                            if value.dtype == np.dtype("O"):
                                dtype = h5py.string_dtype()
                            try:
                                hdf_dataset = group.create_dataset(
                                    dataset_name,
                                    data=value,
                                    compression=dataset_compression,
                                    dtype=dtype,
                                    **dataset_kwargs,
                                )
                            except TypeError:
                                # TODO
                                # print(f"Cannot save array {value} to HDF, skipping it...")
                                pass
                    else:
                        if dataset_name not in group:
                            raise ValueError(
                                f"Dataset {dataset_name} does not exist in group "
                                f"{group_name} of {self}."
                            )
                        if isinstance(value, pd.DataFrame):
                            new_group_name = f"{group_name}/{dataset_name}"
                            if not self.read(
                                group_name=new_group_name,
                                attr_name="is_pd_dataframe",
                            ):
                                raise ValueError(
                                    f"Dataset {dataset_name} in group"
                                    f"{group_name} of {self} is not a dataframe."
                                )
                            existing_columns = self.read(group_name=new_group_name)
                            for col in value.columns:
                                if col not in existing_columns:
                                    raise KeyError(
                                        f"Column {col} is missing in the original dataset {dataset_name} in group"
                                        f"{group_name} of {self}."
                                    )
                            for col in existing_columns:
                                if col not in value.columns:
                                    raise KeyError(
                                        f"Column {col} is missing in the dataset to append to {dataset_name} in group"
                                        f"{group_name} of {self}."
                                    )
                            dtypes = self.read(
                                group_name=group_name,
                                dataset_name=dataset_name,
                                return_dataset_dtype=True,
                            )
                            for col, dtype in zip(existing_columns, dtypes):
                                if value.dtypes[col] != dtype and dtype != np.dtype(
                                    "O"
                                ):
                                    raise TypeError(
                                        f"Types of column {col} do not match between the appended dataset ({value.dtypes[col]}) and {dataset_name} ({dtype}) in group"
                                        f"{group_name} of {self}."
                                    )
                            for column in value.columns:
                                self.write(
                                    value[column].values,
                                    group_name=new_group_name,
                                    dataset_name=column,
                                    overwrite=overwrite,
                                    append=append,
                                    dataset_compression=dataset_compression,
                                    **dataset_kwargs,
                                )
                        else:
                            dtype = value.dtype
                            if value.dtype == np.dtype("O"):
                                dtype = h5py.string_dtype()
                            hdf_dataset = group[dataset_name]
                            hdf_dataset.resize(
                                hdf_dataset.shape[0] + len(value), axis=0
                            )
                            hdf_dataset[-len(value) :] = value
                else:
                    try:
                        dataset = group[dataset_name]
                    except KeyError:
                        raise KeyError(
                            f"Dataset {dataset_name} does not exist for "
                            f"group {group_name} of {self}."
                        )
                    if (attr_name in dataset.attrs) and not overwrite:
                        raise ValueError(
                            f"Attribute {attr_name} already exists in "
                            f"dataset {dataset_name} of group "
                            f"{group_name} of {self}."
                        )
                    try:
                        dataset.attrs[attr_name] = value
                    except TypeError:
                        dataset.attrs[attr_name] = str(value)  # e.g. dicts
            hdf_file.attrs["last_updated"] = time.asctime()

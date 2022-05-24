from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd


def associate_rev_id_to_its_images(
    id: str, path_to_slices: Path, nb_images: int, relative_path: Path = None
) -> List[Path]:
    """Associates a rev id, like `Spec-5` to its sliced images, given a number of slices per plane and a path to slices.

    Args:
        id (str): id of the rev id
        path_to_slices (Path): path to the slices images
        nb_images (int): number of slices per plane
        relative_path (Path): If not None, the output paths will all be relative to `relative_path`.
            Defaults to `None`.

    Returns:
        List[Path]: list of path where each path points an image of the rev `id`.
    """
    return [
        str(x.relative_to(relative_path)) if relative_path is not None else str(x)
        for x in path_to_slices.glob(f"{nb_images}p*/{id}_Imgs/*")
    ]


def convert_into_single_entry_df(
    multi_entry_df: pd.DataFrame, col_name: str = "photos"
) -> pd.DataFrame:
    """Converts a column of a dataframe whose cells are list, into a longer dataframe where the cells are now values of the list
    Supposing `multi_entry_df` looks like this:

    +---+-----------------+
    |   | col_name        |
    +---+-----------------+
    | 1 | ["a", "b", "c"] |
    +---+-----------------+
    | 2 | ["x", "y", "z"] |
    +---+-----------------+

    The output will be:

    +---+----------+
    |   | col_name |
    +---+----------+
    | 1 | "a"      |
    +---+----------+
    | 2 | "x"      |
    +---+----------+
    | 3 | "b"      |
    +---+----------+
    | 4 | "y"      |
    +---+----------+
    | 5 | "c"      |
    +---+----------+
    | 6 | "z"      |
    +---+----------+

    Each list in `multi_entry_df[col_name]` must have the same length.
    Args:
        multi_entry_df (pd.DataFrame): dataframe to modify.
        col_name (str): name of the column in the dataframe whose cells are list

    Raises:
        ValueError: Each list in `multi_entry_df[col_name]` must have the same length.

    Returns:
        pd.DataFrame: copy of the dataframe, and `dataframe[col_name]` are now values of the list
    """

    if not (multi_entry_df[col_name].apply(type) == list).any():
        return multi_entry_df
    nb_images = multi_entry_df[col_name].str.len().max()
    if nb_images != multi_entry_df[col_name].str.len().min():
        raise ValueError(
            "Each list in multi_entry_df[col_name] must have the same length."
        )
    single_entry_df = pd.concat([multi_entry_df] * nb_images, ignore_index=True)
    single_entry_df[col_name] = single_entry_df.apply(
        func=lambda row: str(row[col_name][row.name % nb_images]),
        axis=1,
    )
    return single_entry_df


def get_path_image_along_axis(
    list_paths: List[Union[str, Path]], axis: str = "x"
) -> List[Union[str, Path]]:
    """Keeps only the image paths along a specified axis.

    Example:
        list_paths = ["REV1/x-y(1).png", "REV1/x-y(2).png", "REV1/y-z(1).png", "REV1/y-z(2).png", "REV1/z-x(1).png", "REV1/z-x(2).png",]
        get_path_image_along_axis(list_paths, axis=y)
        >>> ["REV1/z-x(1).png", "REV1/z-x(2).png",]

    Args:
        list_paths (List[Union[str, Path]]): list whose values are the image paths.
            We suppose the image paths is `/.../y-z[XXX].png` if the image is taken along x axis.
        axis (str, optional): axis to keep. Defaults to "x".

    Returns:
        List[Union[str, Path]]: paths of sliced images taken along `axis`
    """
    if isinstance(list_paths, list):
        return [s for s in list_paths if axis not in Path(s).stem]
    else:
        return axis not in Path(list_paths).stem


def _compute_photos_along_axis(
    df: pd.DataFrame, axis: str, col_name: str = "photos", nb_image_per_axis: int = 1
) -> np.ndarray:
    """Do the following:
    * Takes all the image paths in `df[col_name]`
    * selects only the images taken along the `axis`
    * groups all the images of the same REV into groups of `nb_image_per_axis`. Thus, it can have more groups than REV

    Example:
        Let's suppose `df[col_name]` looks like this:

        +---+-----------------------------------------------------------+
        |   | col_name                                                  |
        +---+-----------------------------------------------------------+
        | 1 | ["REV1/x-y(1).png", "REV1/x-y(2).png", "REV1/x-y(3).png"  |
        |   |  "REV1/y-z(1).png", "REV1/y-z(2).png", "REV1/y-z(3).png"  |
        |   |  "REV1/z-x(1).png", "REV1/z-x(2).png", "REV1/z-x(3).png"] |
        +---+-----------------------------------------------------------+
        | 2 | ["REV2/x-y(1).png", "REV2/x-y(2).png", "REV2/x-y(3).png"  |
        |   |  "REV2/y-z(1).png", "REV2/y-z(2).png", "REV2/y-z(3).png"  |
        |   |  "REV2/z-x(1).png", "REV2/z-x(2).png", "REV2/z-x(3).png"] |
        +---+-----------------------------------------------------------+

        Then, `_compute_photos_along_axis(df, axis="z", col_name=col_name, nb_image_per_axis=2)` is a np.ndarray:

        [["REV1/z-x(1).png", "REV1/z-x(2).png"],
         ["REV2/z-x(1).png", "REV2/z-x(2).png"]]

        Initially, there are 3 sliced images per axis, but as we are asking to form groups of 2 images per axis,
        we throw away the last image on the y-axis.


    Args:
        df (pd.DataFrame): _description_
        axis (str): _description_
        col_name (str, optional): _description_. Defaults to "photos".
        nb_image_per_axis (int, optional): _description_. Defaults to 1.

    Raises:
        ValueError: Each list in df[col_name] must have the same length.

    Returns:
        np.ndarray: array containing the lists of length `nb_image_per_axis` of sliced images on `axis`.
    """
    if df[col_name].apply(len).min() != df[col_name].apply(len).max():
        raise ValueError("Each list in df[col_name] must have the same length.")
    nb_photos_per_plane = df[col_name].apply(len).unique().min() // 3
    photos = np.hsplit(
        df[col_name]
        .apply(func=get_path_image_along_axis, args=(axis))
        .explode()
        .to_numpy()
        .reshape(-1, nb_photos_per_plane),
        list(range(0, nb_photos_per_plane, nb_image_per_axis)),
    )[1:]
    return np.concatenate(
        [x for x in photos if x.shape[1] == photos[0].shape[1]], axis=0
    )


def convert_into_n_entry_df(
    multi_entry_df: pd.DataFrame,
    col_name: str = "photos",
    nb_image_per_axis: int = 1,
    order: str = "xyz",
) -> pd.DataFrame:
    """Converts a column of a dataframe whose cells are list of image paaths,
        into a longer dataframe where the cells list of lenght `3*nb_image_per_axis`, following the `order`.

    Example:
        Let's suppose `multi_entry_df[col_name]` looks like this:

        +---+-----------------------------------------------------------+
        |   | col_name                                                  |
        +---+-----------------------------------------------------------+
        | 1 | ["REV1/x-y(1).png", "REV1/x-y(2).png", "REV1/x-y(3).png"  |
        |   |  "REV1/y-z(1).png", "REV1/y-z(2).png", "REV1/y-z(3).png"  |
        |   |  "REV1/z-x(1).png", "REV1/z-x(2).png", "REV1/z-x(3).png"] |
        +---+-----------------------------------------------------------+
        | 2 | ["REV2/x-y(1).png", "REV2/x-y(2).png", "REV2/x-y(3).png"  |
        |   |  "REV2/y-z(1).png", "REV2/y-z(2).png", "REV2/y-z(3).png"  |
        |   |  "REV2/z-x(1).png", "REV2/z-x(2).png", "REV2/z-x(3).png"] |
        +---+-----------------------------------------------------------+

        Then, `convert_into_n_entry_df(df, col_name=col_name, nb_image_per_axis=2, order="yzx")[col_name]` is:

        +---+-----------------------------------------------------------+
        |   | col_name                                                  |
        +---+-----------------------------------------------------------+
        | 1 | ["REV1/z-x(1).png", "REV1/z-x(2).png",                    |
        |   |  "REV1/x-y(1).png", "REV1/x-y(2).png",                    |
        |   |  "REV1/y-z(1).png", "REV1/y-z(2).png"]                    |
        +---+-----------------------------------------------------------+
        | 2 | ["REV2/z-x(1).png", "REV2/z-x(2).png",                    |
        |   |  "REV2/x-y(1).png", "REV2/x-y(2).png",                    |
        |   |  "REV2/y-z(1).png", "REV2/y-z(2).png"]                    |
        +---+-----------------------------------------------------------+

    Args:
        multi_entry_df (pd.DataFrame): dataframe to modify
        col_name (str, optional): column's name containing the image paths. Defaults to "photos".
        nb_image_per_axis (int, optional): nb of images per axis to take in each group. Defaults to 1.
        order (str, optional): order of the groups. Defaults to "xyz".

    Raises:
        ValueError: Each cell in `multi_entry_df[col_name]` must have at least 3*nb_image_per_axis elements

    Returns:
        pd.DataFrame: modified dataframe
    """
    if not (multi_entry_df[col_name].apply(len) >= 3 * nb_image_per_axis).all():
        raise ValueError(
            "Each cell in multi_entry_df[col_name] must have at least 3*nb_image_per_axis elements."
        )
    x_photos = _compute_photos_along_axis(
        multi_entry_df, "x", nb_image_per_axis=nb_image_per_axis
    )
    y_photos = _compute_photos_along_axis(
        multi_entry_df, "y", nb_image_per_axis=nb_image_per_axis
    )
    z_photos = _compute_photos_along_axis(
        multi_entry_df, "z", nb_image_per_axis=nb_image_per_axis
    )

    if order == "random":
        images = np.concatenate([x_photos, y_photos, z_photos], axis=1)
        images = np.apply_along_axis(np.random.permutation, axis=1, arr=images)

    else:
        index_order = ["xyz".index(carac) for carac in order]
        unordered_images = [x_photos, y_photos, z_photos]
        ordered_images = [unordered_images[i] for i in index_order]
        images = np.concatenate(ordered_images, axis=1)

    df = pd.concat(
        [multi_entry_df] * (len(x_photos) // len(multi_entry_df)),
        ignore_index=True,
    )
    df[col_name] = images.tolist()
    return df


def convert_into_n_width_df(
    multi_entry_df: pd.DataFrame,
    col_name: str = "photos",
    nb_image_per_axis: int = 4,
    order: str = "xyz",
):
    """Convert a dataframe into a n witdh dataframe.

    Example:
        Supposing `multi_entry_df[col_name]` looks like this:

        +---+-----------------------------------------+
        |   | col_name                                |
        +---+-----------------------------------------+
        | 1 | ["REV1/x-y(1).png", "REV1/x-y(2).png",  |
        |   |  "REV1/y-z(1).png", "REV1/y-z(2).png",  |
        |   |  "REV1/z-x(1).png", "REV1/z-x(2).png",] |
        +---+-----------------------------------------+
        | 2 | ...                                     |
        +---+-----------------------------------------+

        We transform it into:

        +---+-------------------------------------------+
        |   | col_name                                  |
        +---+-------------------------------------------+
        | 1 | [["REV1/x-y(1).png", "REV1/x-y(2).png"],  |
        |   |  ["REV1/y-z(1).png", "REV1/y-z(2).png"],  |
        |   |  ["REV1/z-x(1).png", "REV1/z-x(2).png"]]  |
        +---+-------------------------------------------+
        | 2 | ...                                       |
        +---+-------------------------------------------+

        where images along x, y and Z all belongs to the same list. There are `nb_image_per_axis` images on x/y/z axis per rev.
        If there are enough photos, we can create more rev.

    Args:
        multi_entry_df (pd.DataFrame): dataframe to modify
        col_name (str, optional): column name where the photos are located. Defaults to "photos".
        nb_image_per_axis (int, optional): number of input photos per plane. Defaults to 4.
        order (str, optional): order of the photos. Defaults to "xyz".

    Raises:
        ValueError: Each cell in `multi_entry_df[col_name]` must have at least 3*nb_image_per_axis elements
        NotImplementedError: if the order is "random"

    Returns:
        pd.DataFrame: multi_entry_df where `multi_entry_df[col_name]` has been updated
    """
    if not (multi_entry_df[col_name].apply(len) >= 3 * nb_image_per_axis).all():
        raise ValueError(
            "Each cell in multi_entry_df[col_name] must have at least 3*nb_image_per_axis elements."
        )
    x_photos = _compute_photos_along_axis(
        multi_entry_df, "x", nb_image_per_axis=nb_image_per_axis
    )
    y_photos = _compute_photos_along_axis(
        multi_entry_df, "y", nb_image_per_axis=nb_image_per_axis
    )
    z_photos = _compute_photos_along_axis(
        multi_entry_df, "z", nb_image_per_axis=nb_image_per_axis
    )

    if order == "random":
        raise NotImplementedError("The order can't be `random`.")
    else:
        index_order = ["xyz".index(carac) for carac in order]
        unordered_images = [x_photos, y_photos, z_photos]
        images = np.stack([unordered_images[i] for i in index_order], axis=1)

    df = pd.concat(
        [multi_entry_df] * (len(x_photos) // len(multi_entry_df)),
        ignore_index=True,
    )
    df[col_name] = images.tolist()
    return df

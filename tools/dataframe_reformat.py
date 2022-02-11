from pathlib import Path
from typing import List, Union

import pandas as pd


def associate_rev_id_to_its_images(
    id: str, path_to_slices: Path, nb_images: int, relative_path: Path = None
) -> List[Path]:
    """Associates a rev id, like `Spec-5` to its sliced images, given a number of slices per plane and

    Args:
        id (str): id of the rev id
        path_to_slices (Path): path to the slices images
        nb_images (int): number of slices per plane

    Returns:
        List[Path]: list of path where each path points an image
    """
    return [
        str(x.relative_to(relative_path)) if relative_path is not None else str(x)
        for x in path_to_slices.glob(f"{nb_images}p*/{id}_Imgs/*")
    ]


def convert_into_single_entry_df(
    multi_entry_df: pd.DataFrame, col_name: str = "photos"
) -> pd.DataFrame:
    """Converts a column of a dataframe whose cells are list, into a longer dataframe where the cells are now values of the list

    Args:
        multi_entry_df (pd.DataFrame): dataframe to modify
        col_name (str): name of the column in the dataframe whose cells are list

    Returns:
        pd.DataFrame: copy of the dataframe, and `dataframe[col_name]` are now values of the list
    """
    if not (multi_entry_df[col_name].apply(type) == list).any():
        return multi_entry_df
    nb_images = multi_entry_df[col_name].str.len().max()
    assert nb_images == multi_entry_df[col_name].str.len().min()
    single_entry_df = pd.concat([multi_entry_df] * nb_images, ignore_index=True)
    single_entry_df[col_name] = single_entry_df.apply(
        func=lambda row: str(row[col_name][row.name % nb_images]),
        axis=1,
    )
    return single_entry_df


def get_path_image_along_axis(
    l: List[Union[str, Path]], axis: str = "x"
) -> List[Union[str, Path]]:
    """Keeps only the image paths along a specified axis

    Args:
        l (List[Union[str, Path]]): list whose values are the image paths.
            We suppose the image filename is "y-z[XXX].png" if the image is taken along x axis.
        axis (str, optional): axis to keep. Defaults to "x".

    Returns:
        List[Union[str, Path]]: paths of sliced images taken along `axis`
    """
    return [s for s in l if axis not in Path(s).stem]

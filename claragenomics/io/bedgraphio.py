#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

"""Modules to write to bedGraph format."""

# Import requirements
import pandas as pd
import cudf
from numba import cuda
import numpy as np
import sys

def add_chrom(interval, score=True):
    """Expand an interval to single-base resolution and add scores.

    Args:
        interval : dict or DataFrame containing chrom, start, end and
        optionally scores.
        score : Boolean to specify whether to add score to each base of
        interval

    Returns:
        expanded: pandas DataFrame containing a row for every base in the
        interval

    """
    chrom_add = pd.DataFrame()
    size = interval['end'] - interval['start']
    chrom_add['chrom'] = [interval['chrom']]*size
    return chrom_add


@cuda.jit
def expand_interval(in_col, in_col2, out_col, out_col2):
    i = cuda.grid(1)
    if i < in_col.size: # boundary guard
        for j in range(in_col[i], in_col2[i]):
            out_col[j] = j
            out_col2[j] = j + 1


@cuda.jit
def get_diff(in1, diff):
    i = cuda.grid(1)
    if i < in1.size: # boundary guard
        if i == 0:
            diff[i] = 2
        else:
            diff[i] = in1[i] - in1[i-1]


def intervals_to_bg(intervals_df, mode):
    """Format intervals + scores to bedGraph format.

    Args:
        intervals_df: Pandas dataframe containing columns for chrom, start,
        end and scores.

    Returns:
        bg: pandas dataframe containing expanded+contracted intervals.

    """
    if mode == "expand":
        # Add chromosome column separately because cudf doesn't support
        # string operations.
        chrom_df = intervals_df.apply(add_chrom, axis = 1)
        chrom_df = pd.concat(list(chrom_df), ignore_index=True)
        # Convert pandas df to cuda and expand intervals.
        intervals_df_cuda = cudf.from_pandas(intervals_df)
        df = cudf.DataFrame()
        size = intervals_df_cuda['end'][-1] - intervals_df_cuda['start'][0]
        df['start'] = np.zeros(size, dtype=np.int64)
        df['end'] = np.zeros(size, dtype=np.int64)
        expand_interval.forall(size)(intervals_df_cuda['start'],
                            intervals_df_cuda['end'],
                            df['start'],
                            df['end'])

        # Convert the df back to pandas and join the chrom df.
        bg = df.to_pandas()
        bg['chrom'] = chrom_df['chrom']
        return bg
    elif mode == "contract":
        # Create a new column with scores moved one row up and calculate diff
        # between new scores and original scores. Rows where diff is non-zero 
        # will be rows where score has changed.

        intervals_df["diff"] = 0

        cudf_file = cudf.from_pandas(intervals_df)
        size = len(cudf_file['scores'])
        get_diff.forall(size)(cudf_file['scores'], cudf_file['diff'])

        #Choose cols with diff > 0
        less_bg = cudf_file[(cudf_file['diff'] != 0) | (cudf_file.index == len(cudf_file) - 1)].copy()

        # Modify end column.
        less_bg['end'] = list(less_bg['start'])[1:] + \
                                [less_bg['end'].iloc[-1]]

        # Keep scores > 0 and keep relevant cols
        bg = less_bg.loc[:, ['chrom', 'start', 'end', 'scores']]
        bg = bg.to_pandas()
        return bg


def df_to_bedGraph(df, outfile, sizes=None):
    """Write a dataframe in bedGraph format to a bedGraph file.

    Args:
        df : dataframe to be written.
        outfile : file name or object.
        sizes: dataframe containing chromosome sizes.

    """
    if sizes is not None:
        # Write only entries for the given chromosomes.
        num_drop = sum(~df['chrom'].isin(sizes['chrom']))
        print("Discarding " + str(num_drop) + " entries outside sizes file.")
        df = df[df['chrom'].isin(sizes['chrom'])]
        # Check that no entries exceed chromosome lengths.
        df_sizes = df.merge(sizes, on='chrom')
        excess_entries = df_sizes[
            df_sizes['end'] > df_sizes['length']]
        assert len(excess_entries) == 0, \
            "Entries exceed chromosome sizes ({})".format(excess_entries)
    assert len(df) > 0, "0 entries to write to bedGraph"
    df.to_csv(outfile, sep='\t', header=False, index=False)

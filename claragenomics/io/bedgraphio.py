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
import math

# GLOBAL DECLARATION

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
def expand_interval(in_col, in_col2, out_col, out_col2, interval_size):
    i = cuda.grid(1)
    if i < out_col.size: # boundary guard
        batch_id = int(math.floor(i/interval_size))
        out_col[i] = in_col[batch_id] + i%interval_size
        out_col2[i] = out_col[i] + 1

@cuda.jit
def get_prev_score(in1, prev, interval_size):
    i = cuda.grid(1)
    if i < in1.size: # boundary guard
        if i%interval_size == 0 :
            prev[i] = -1
        else:
            prev[i] = in1[i-1]

@cuda.jit
def modify_end(start, end, interval_size):
    i = cuda.grid(1)
    if i < start.size-1: # boundary guard
        if start[i+1]%interval_size != 0 :
            end[i] = start[i+1]


def intervals_to_bg(intervals_df, mode, batch_size):
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
        intervals_df_pd = intervals_df.to_pandas()
        chrom_df = intervals_df_pd.apply(add_chrom, axis = 1)
        chrom_df = pd.concat(list(chrom_df), ignore_index=True)
        # Convert pandas df to cuda and expand intervals.
        interval_size = intervals_df['end'][0] - intervals_df['start'][0]
        size = batch_size*interval_size
        df = cudf.DataFrame()
        df['start'] = np.zeros(size, dtype=np.int32)
        df['end'] = np.zeros(size, dtype=np.int32)
        expand_interval.forall(size)(intervals_df['start'],
                            intervals_df['end'],
                            df['start'],
                            df['end'],
                            interval_size)

        # Convert the df back to pandas and join the chrom df.
        df.insert(0, 'chrom', chrom_df['chrom'])

        return df
    elif mode == "contract":
        # Create a new column with scores moved one row up and calculate diff
        # between new scores and original scores. Rows where diff is non-zero 
        # will be rows where score has changed.

        intervals_df["prev_score"] = 0.0
        size = len(intervals_df['scores'])
        interval_size = int(size/batch_size)
        get_prev_score.forall(size)(intervals_df['scores'], intervals_df['prev_score'], interval_size)
        #Choose cols with diff > 0
        pd_file = intervals_df.to_pandas()
        #intervals_df.to_csv("contract_df.bedGraph", sep='\t', header=None)
        less_bg = pd_file.loc[pd_file["scores"] != pd_file["prev_score"],:].copy()

        less_bg = cudf.from_pandas(less_bg)
        # Modify end column.
        size = len(less_bg['start'])
        modify_end.forall(size)(less_bg['start'], less_bg['end'], interval_size)

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

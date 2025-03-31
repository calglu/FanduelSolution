#This code is a streaming Bayesian approach to find the mode in a large dataset.
#Used co-pilot for naming conventions, docstrings, and diagnostics

import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from pybloom_live import BloomFilter
from collections import Counter


##### User-defined Parameters
PRUNE_MODE_PROB_THRESHOLD = 0.01 #Threshold for pruning rows - 1%
TOTAL_ROWS_GIVEN = 1000000 #Total number of rows in the dataset
PRIOR_WEIGHT = 0.1
UNSEEN_WEIGHT = 1.0
TOP_K_PROTECTED = 5 #Number of top-K rows to protect from pruning
CONFIDENCE_CUTOFF = 0.999 #Confidence threshold for early exit - 99.9%
CONFIDENCE_STREAK = 3 #Number of consecutive times to hit CONFIDENCE_CUTOFF before exiting

#alpha for Dirichlet smoothing of known categories
ALPHA = 1.0

#Auto-scaled Parameters
SAMPLE_SIZE = max(int(0.01 * TOTAL_ROWS_GIVEN), 1000)
MAX_ROWS_TRACKED = max(int(0.001 * TOTAL_ROWS_GIVEN), 500) #Maximum number of rows in tracker
BLOOM_CAPACITY = max(TOTAL_ROWS_GIVEN, 10000)
BLOOM_ERROR_RATE = 0.001

#Globals
bloom = BloomFilter(capacity=BLOOM_CAPACITY, error_rate=BLOOM_ERROR_RATE)
row_counter = Counter()
prior_counts = Counter()
total_unique_seen = 0

#Diagnostics - removing may improve memory performance
prune_log = []
topk_log = []
prob_log = []
time_log = []
confidence_history = []


def normalize_row(row):
    """Clean or parse the row string as needed."""
    return row.strip()


def estimate_p_unseen_good_turing(counter):
    """
    Good-Turing estimate: P_unseen = N1 / N,
      N1 = #categories observed exactly once,
      N  = total #observations.
    """
    freq_of_freqs = Counter(counter.values())
    f1 = freq_of_freqs.get(1, 0)
    total = sum(counter.values())
    if total == 0:
        return 1.0 #If no data at all, everything is unseen
    return f1 / total


def sample_prior(filepath, sample_size=SAMPLE_SIZE):
    """
    Prior sample, which will be used for posterior updates.
    """
    counts = Counter()
    with open(filepath, 'r') as file:
        for i, line in enumerate(file):
            if i >= sample_size:
                break
            row = normalize_row(line)
            counts[row] += 1
    return counts


def compute_posterior_single_unseen(prior_counts, observed_counts, alpha=ALPHA):
    """
    Computes a dictionary: {category -> pseudo-count} with a single __UNSEEN__ bucket.
    Steps:
      1. Summation of known categories:
           posterior[row] = observed_counts[row] + PRIOR_WEIGHT*prior_counts[row] + alpha
      2. Good-Turing to find P_unseen = N1 / N over observed_counts.
      3. __UNSEEN__ mass = p_unseen * (sum of above pseudo-counts) * UNSEEN_WEIGHT.
    """
    from collections import defaultdict

    #Build known categories
    posterior = {}
    all_rows = set(prior_counts.keys()) | set(observed_counts.keys())
    sum_known = 0.0
    for r in all_rows:
        pseudo_count = (observed_counts.get(r, 0)
                        + PRIOR_WEIGHT * prior_counts.get(r, 0)
                        + alpha)
        posterior[r] = pseudo_count
        sum_known += pseudo_count

    #Good-Turing
    p_unseen = estimate_p_unseen_good_turing(observed_counts)

    #Single unseen placeholder
    unseen_mass = max(p_unseen * sum_known * UNSEEN_WEIGHT, 1e-9)
    posterior["__UNSEEN__"] = unseen_mass

    return posterior


def estimate_mode_probabilities_single_unseen(posterior_counts, num_samples=250):
    """
    Approximates mode probabilities among known categories:
      1) Creates Dirichlet(alpha=posterior_counts).
      2) Draws samples, ignoring "__UNSEEN__" as a potential mode.
      3) Returns {category -> P(mode)} for known categories.
    """
    rows = list(posterior_counts.keys())
    alpha_vector = np.array([posterior_counts[r] for r in rows], dtype=float)

    samples = np.random.dirichlet(alpha_vector, size=num_samples) #Sample from Dirichlet

    #Identify index of unseen dimension
    try:
        idx_unseen = rows.index("__UNSEEN__")
    except ValueError:
        idx_unseen = -1

    mode_counts = Counter()
    for s in samples:
        #Temporarily zero out the unseen dimension so it can't win
        local_s = s.copy()
        if idx_unseen >= 0:
            local_s[idx_unseen] = -1  # ensures it won't be argmax
        max_idx = np.argmax(local_s)
        mode_counts[rows[max_idx]] += 1

    #Convert tallies to probabilities for known categories
    mode_probs = {}
    for r in rows:
        if r == "__UNSEEN__":
            continue
        mode_probs[r] = mode_counts[r] / num_samples

    return mode_probs


def should_prune_by_mode_prob(row, mode_probs):
    """Decide if 'row' should be pruned based on its posterior mode probability."""
    return mode_probs.get(row, 0.0) < PRUNE_MODE_PROB_THRESHOLD


def find_most_frequent_row(filepath):
    """
    Streams through the data, updating row_counter for each line after
    the initial prior sample. Periodically:
      1) Recomputes single-unseen posterior
      2) Estimates mode probabilities
      3) Prunes categories with low probability of being the mode
    If a category's P(mode) hits CONFIDENCE_CUTOFF for CONFIDENCE_STREAK times,
    we exit early for efficiency.
    """
    global total_unique_seen
    total_rows_seen = 0

    #For adjusting how often we do Dirichlet updates
    dirichlet_interval = max(int(0.001 * TOTAL_ROWS_GIVEN), 100) #Initial interval for Dirichlet updates
    next_dirichlet_update = SAMPLE_SIZE + dirichlet_interval
    growth_factor = 1.5 #Growth factor for Dirichlet update interval
    max_interval = max(int(0.05 * TOTAL_ROWS_GIVEN), 5000) #Max interval for Dirichlet updates

    with open(filepath, 'r') as file:
        for i, line in enumerate(file):
            #Skip lines used in the prior
            if i < SAMPLE_SIZE:
                continue

            total_rows_seen += 1
            row = normalize_row(line)

            #Bloom filter for unique row counting
            if row not in bloom:
                bloom.add(row)
                total_unique_seen += 1

            #Track row if within capacity
            if row in row_counter:
                row_counter[row] += 1
            elif len(row_counter) < MAX_ROWS_TRACKED:
                row_counter[row] = 1

            #Periodic posterior update + prune
            if (len(row_counter) > MAX_ROWS_TRACKED * 0.9) and (total_rows_seen >= next_dirichlet_update):
                t0 = time.time()

                #Build single-unseen posterior
                posterior = compute_posterior_single_unseen(prior_counts, row_counter, alpha=ALPHA)
                # Estimate mode probabilities
                mode_probs = estimate_mode_probabilities_single_unseen(posterior, num_samples=250)

                #Protect top-K frequent rows from pruning
                top_rows = {r for r, _ in row_counter.most_common(TOP_K_PROTECTED)}
                prune_count = 0
                for tracked_row in list(row_counter):
                    if tracked_row not in top_rows and should_prune_by_mode_prob(tracked_row, mode_probs):
                        del row_counter[tracked_row]
                        prune_count += 1

                #Logging
                t1 = time.time()
                prune_log.append((total_rows_seen, prune_count))
                time_log.append(t1 - t0)

                if mode_probs:
                    top_mode = max(mode_probs.items(), key=lambda x: x[1])
                    prob_log.append((total_rows_seen, top_mode[0], top_mode[1]))

                    # Early-exit confidence
                    if top_mode[1] >= CONFIDENCE_CUTOFF:
                        confidence_history.append(top_mode[0])
                    else:
                        confidence_history.clear()

                    if (len(confidence_history) >= CONFIDENCE_STREAK 
                        and len(set(confidence_history)) == 1):
                        print(f"[Early Exit] High confidence in mode '{top_mode[0]}' "
                              f"(P={top_mode[1]:.4f}) at row {total_rows_seen}")
                        break

                topk_log.append(list(top_rows))

                #Adjust next update interval
                dirichlet_interval = min(int(dirichlet_interval * growth_factor), max_interval)
                next_dirichlet_update = total_rows_seen + dirichlet_interval

    #Final posterior update after streaming
    if not row_counter:
        return None, {}

    final_posterior = compute_posterior_single_unseen(prior_counts, row_counter, alpha=ALPHA)
    print(f"[End-of-Stream] Using Good-Turing single-unseen posterior. Counted rows: {sum(row_counter.values())}")
    final_mode_probs = estimate_mode_probabilities_single_unseen(final_posterior, num_samples=500)

    if final_mode_probs:
        most_likely = max(final_mode_probs.items(), key=lambda x: x[1])
    else:
        most_likely = (None, 0.0)

    return most_likely, final_mode_probs


def streaming_true_mode(filepath, sample_skip=SAMPLE_SIZE):
    """
    Simple "true mode" reference after skipping 'sample_skip' lines.
    Just a naive frequency count to see which row truly appears most often.
    """
    import tracemalloc
    simple_counter = Counter()

    tracemalloc.start()
    t0 = time.time()

    with open(filepath, 'r') as file:
        for i, line in enumerate(file):
            if i < sample_skip:
                continue
            row = normalize_row(line)
            simple_counter[row] += 1

    t1 = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if not simple_counter:
        return None, 0.0, (t1 - t0), peak

    top_row, count = simple_counter.most_common(1)[0]
    total = sum(simple_counter.values())
    return top_row, count / total, (t1 - t0), peak


if __name__ == "__main__":

    filepath = "./projects/FanduelTakeHome/large_dataset.txt"

    #Sample from the dataset for a prior
    prior_counts = sample_prior(filepath, sample_size=SAMPLE_SIZE)

    tracemalloc.start()
    start_time = time.time()

    #Run the whole thing
    result, mode_probs = find_most_frequent_row(filepath)

    end_time = time.time()
    current, peak_bayes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if result and result[0] is not None:
        print(f"\n[Bayesian] Most likely mode: {result[0]}")
        print(f"[Bayesian] Estimated probability: {result[1]:.4f}")
    else:
        print("[Bayesian] No most frequent row found.")

    print(f"[Bayesian] Time elapsed: {end_time - start_time:.2f} seconds")
    print(f"[Bayesian] Peak memory usage: {peak_bayes / 1024:.2f} KB")

    #Simple streaming approach
    streaming_row, streaming_prob, streaming_time, peak_stream = streaming_true_mode(filepath)

    print(f"\n[Streaming] Most frequent row: {streaming_row}")
    print(f"[Streaming] True probability: {streaming_prob:.4f}")
    print(f"[Streaming] Time elapsed: {streaming_time:.2f} seconds")
    print(f"[Streaming] Peak memory usage: {peak_stream / 1024:.2f} KB")

    print(f"\nFinal tracked rows (Bayesian): {len(row_counter)}")
    print(f"Top 10 rows by count (Bayesian): {row_counter.most_common(10)}")
    print(f"Unique rows seen (estimated): {total_unique_seen}")

    #Plotting Diagnostics - Copilot
    if prune_log:
        x_vals, prune_vals = zip(*prune_log)
        plt.figure(figsize=(10, 4))
        plt.plot(x_vals, prune_vals, label="Rows pruned")
        plt.xlabel("Rows seen")
        plt.ylabel("Pruned rows")
        plt.title("Pruning events over time")
        plt.legend()
        plt.grid(True)
        plt.show()

    if prob_log:
        x_vals, labels, probs = zip(*prob_log)
        plt.figure(figsize=(10, 4))
        plt.plot(x_vals, probs, label="P(mode)")
        plt.xlabel("Rows seen")
        plt.ylabel("Estimated P(mode)")
        plt.title("Estimated probability of most likely mode")
        plt.grid(True)
        plt.legend()
        plt.show()

    if time_log:
        plt.figure(figsize=(10, 4))
        plt.plot(range(len(time_log)), time_log, label="Dirichlet update time (s)")
        plt.xlabel("Dirichlet step")
        plt.ylabel("Seconds")
        plt.title("Time taken per Dirichlet update")
        plt.grid(True)
        plt.legend()
        plt.show()

    #Save diagnostics
    import pickle
    with open("diagnostics.pkl", "wb") as f:
        pickle.dump({
            "prob_log": prob_log,
            "time_log": time_log
        }, f)

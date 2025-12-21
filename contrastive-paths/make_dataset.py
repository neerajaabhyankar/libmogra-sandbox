from imports import *
from utils import ratio_to_coord, encode_hf, create_harmono_frequency_space_paths, add_to_dataset


def quick_test():
    primes = [3, 5]
    ef = EFGenus(primes=primes, powers=[4, 1])
    tn = Tonnetz(ef)

    bmpls = [1, Fraction(9,8), Fraction(6, 5), Fraction(4,3), Fraction(3,2), Fraction(27,16), Fraction(9,5)]
    b = ",n S g m P".split(" ")
    b = [SSwar.from_string(s) for s in b]
    samples = create_harmono_frequency_space_paths(b, bmpls, tn)

    assert len(samples["good_paths"]) == 1
    assert len(samples["bad_paths"]) == 47


RAAG_DATASET = {
    "Bheempalasi": {
        "ground_truth": [1, Fraction(9,8), Fraction(6, 5), Fraction(4,3), Fraction(3,2), Fraction(27,16), Fraction(9,5)],
        "phrases": [
            ",n S g m P",
            "n D P",
            "m P g m",
            ",n S g R S",
            "m g R S",
        ],
    },
    "Darbari": {
        "ground_truth": [Fraction(128,81), Fraction(32,27), Fraction(10, 9), Fraction(16,9), Fraction(4,3), Fraction(1), Fraction(3,2), Fraction(40, 27)],
        "phrases": [
            "S R g g",
            "m P d d n P",
            "n n P m P g",
            "m P g g m R S",
            ",n S R S ,d ,d ,n S",
            "`S d d n P",
            "R R S ,n S",
            ",d ,n R",
        ],
    },
    "Puriya Dhanashree": {
        "ground_truth": [Fraction(1), Fraction(3,2), Fraction(5,4), Fraction(15,8), Fraction(45,32), Fraction(135, 128), Fraction(405, 256)],
        "phrases": [
            ",N r G M P",
            "P d M P",
            "M G M r G",
            "M d N `S",
            "N `r N d P",
            "M d M G",
            "N `r `G `r `S",
            "N d N `r N d P",
        ],
    },
    "Bhairav": {
        "ground_truth": [Fraction(1), Fraction(3,2), Fraction(5,4), Fraction(15,8), Fraction(4,3), Fraction(16,15), Fraction(8,5)],
        "phrases": [
            "G m r r S",
            "`S N d d P",
            "G m d d P",
            ",N S r r S",
            "G m P G m",
            "G m N d",
            "d N `S",
        ],
    },
}

SPLIT = {
    "train": ["Bheempalasi", "Darbari", "Puriya Dhanashree"],
    "val": ["Bhairav"],
}

primes = [3, 5]
ef = EFGenus(primes=primes, powers=[4, 1])
tn = Tonnetz(ef)
all_samples_train = []
all_samples_val = []

def add_to_dataset_helper(r_phrases, r_ground_truth, add_to_split=None):
    """Helper function that uses the global tn object"""
    global tn, all_samples_train, all_samples_val
    add_to_dataset(r_phrases, r_ground_truth, tn, all_samples_train, all_samples_val, add_to_split)

for split, raag_names in SPLIT.items():
    for raag_name in raag_names:
        add_to_dataset_helper(RAAG_DATASET[raag_name]["phrases"], RAAG_DATASET[raag_name]["ground_truth"], add_to_split=split)

# assert that each phrase has at least 1 good path
assert all([len(samples["good_paths"]) > 0 for samples in all_samples_train]), "Not all phrases have good paths"


def print_stats(all_samples_train, all_samples_val):
    print(f"Total samples in train: {len(all_samples_train)}")
    print(f"Total samples in val: {len(all_samples_val)}")
    print(f"Total good paths in train: {sum([len(samples['good_paths']) for samples in all_samples_train])}")
    print(f"Total bad paths in train: {sum([len(samples['bad_paths']) for samples in all_samples_train])}")
    print(f"Total good paths in val: {sum([len(samples['good_paths']) for samples in all_samples_val])}")
    print(f"Total bad paths in val: {sum([len(samples['bad_paths']) for samples in all_samples_val])}")


def check_leaks(all_samples_train, all_samples_val):
    """
    Check for data leakage between train and val sets by:
    1. Hashing full paths to find exact duplicates
    2. Checking if paths share the same endpoints
    """

    def path_key(path_nodes):
        """Create canonical hash for a path (full node sequence)"""
        return tuple(map(tuple, path_nodes))

    def endpoints_key(path_nodes):
        """Create hash based on start and end nodes only"""
        return (tuple(path_nodes[0]), tuple(path_nodes[-1]))

    # Collect all paths from train
    train_good_paths = []
    train_bad_paths = []
    for sample in all_samples_train:
        train_good_paths.extend(sample["good_paths"])
        train_bad_paths.extend(sample["bad_paths"])

    # Collect all paths from val
    val_good_paths = []
    val_bad_paths = []
    for sample in all_samples_val:
        val_good_paths.extend(sample["good_paths"])
        val_bad_paths.extend(sample["bad_paths"])

    # Create hash sets for full paths
    train_good_keys = set(path_key(p) for p in train_good_paths)
    train_bad_keys = set(path_key(p) for p in train_bad_paths)
    val_good_keys = set(path_key(p) for p in val_good_paths)
    val_bad_keys = set(path_key(p) for p in val_bad_paths)

    # Create hash sets for endpoints
    train_good_endpoints = set(endpoints_key(p) for p in train_good_paths)
    train_bad_endpoints = set(endpoints_key(p) for p in train_bad_paths)
    val_good_endpoints = set(endpoints_key(p) for p in val_good_paths)
    val_bad_endpoints = set(endpoints_key(p) for p in val_bad_paths)

    print("=" * 80)
    print("DATA LEAKAGE CHECK")
    print("=" * 80)

    print("\n--- Dataset Statistics ---")
    print(f"Train good paths: {len(train_good_paths)} (unique: {len(train_good_keys)})")
    print(f"Train bad paths:  {len(train_bad_paths)} (unique: {len(train_bad_keys)})")
    print(f"Val good paths:   {len(val_good_paths)} (unique: {len(val_good_keys)})")
    print(f"Val bad paths:    {len(val_bad_paths)} (unique: {len(val_bad_keys)})")

    print("\n--- A. EXACT PATH OVERLAP (Full Node Sequence) ---")

    # Good path overlap
    good_overlap = train_good_keys & val_good_keys
    print(f"Good paths in BOTH train and val: {len(good_overlap)}")
    if len(good_overlap) > 0:
        print(f"  -> LEAK DETECTED! {len(good_overlap)} good paths appear in both sets")

    # Bad path overlap
    bad_overlap = train_bad_keys & val_bad_keys
    print(f"Bad paths in BOTH train and val: {len(bad_overlap)}")
    if len(bad_overlap) > 0:
        print(f"  -> LEAK DETECTED! {len(bad_overlap)} bad paths appear in both sets")

    # Cross-contamination: train good = val bad or vice versa
    train_good_val_bad = train_good_keys & val_bad_keys
    train_bad_val_good = train_bad_keys & val_good_keys
    print(f"Train GOOD paths that are Val BAD: {len(train_good_val_bad)}")
    print(f"Train BAD paths that are Val GOOD: {len(train_bad_val_good)}")
    if len(train_good_val_bad) > 0 or len(train_bad_val_good) > 0:
        print(f"  -> LABEL CONFLICT! Same paths have different labels in train vs val")

    print("\n--- B. ENDPOINT OVERLAP (Start/End Nodes Only) ---")

    # Good endpoint overlap
    good_endpoint_overlap = train_good_endpoints & val_good_endpoints
    print(f"Good paths sharing endpoints (train ∩ val): {len(good_endpoint_overlap)}")
    print(f"  Val good paths with train good endpoints: {len([p for p in val_good_paths if endpoints_key(p) in train_good_endpoints])} / {len(val_good_paths)} ({100*len([p for p in val_good_paths if endpoints_key(p) in train_good_endpoints])/len(val_good_paths):.1f}%)")

    # Bad endpoint overlap
    bad_endpoint_overlap = train_bad_endpoints & val_bad_endpoints
    print(f"Bad paths sharing endpoints (train ∩ val): {len(bad_endpoint_overlap)}")
    print(f"  Val bad paths with train bad endpoints: {len([p for p in val_bad_paths if endpoints_key(p) in train_bad_endpoints])} / {len(val_bad_paths)} ({100*len([p for p in val_bad_paths if endpoints_key(p) in train_bad_endpoints])/len(val_bad_paths):.1f}%)")

    # Cross-contamination in endpoints
    train_good_ep_val_bad_ep = train_good_endpoints & val_bad_endpoints
    train_bad_ep_val_good_ep = train_bad_endpoints & val_good_endpoints
    print(f"\nEndpoint conflicts:")
    print(f"  Endpoints that are GOOD in train but BAD in val: {len(train_good_ep_val_bad_ep)}")
    print(f"  Endpoints that are BAD in train but GOOD in val: {len(train_bad_ep_val_good_ep)}")

    print("\n--- VERDICT ---")
    if len(good_overlap) > 0 or len(bad_overlap) > 0:
        print("🔥 SMOKING GUN: Exact path duplicates found between train and val!")
    elif len(good_endpoint_overlap) > 50 or len(bad_endpoint_overlap) > 50:
        print("⚠️  WARNING: Significant endpoint overlap - model may be learning shortcuts!")
    else:
        print("✓ No obvious leakage detected")
    print("=" * 80)


def check_separation(all_samples_train, all_samples_val):
    """
    Check if there are geometric shortcuts the model could exploit:
    1. Nodes that only appear in good or bad paths
    2. Spatial regions correlated with good/bad labels
    """

    def extract_nodes(paths):
        """Extract all unique nodes from a list of paths"""
        nodes = set()
        for path in paths:
            for node in path:
                nodes.add(tuple(node))
        return nodes

    # Collect all paths
    train_good_paths = []
    train_bad_paths = []
    for sample in all_samples_train:
        train_good_paths.extend(sample["good_paths"])
        train_bad_paths.extend(sample["bad_paths"])

    val_good_paths = []
    val_bad_paths = []
    for sample in all_samples_val:
        val_good_paths.extend(sample["good_paths"])
        val_bad_paths.extend(sample["bad_paths"])

    # Extract unique nodes
    train_good_nodes = extract_nodes(train_good_paths)
    train_bad_nodes = extract_nodes(train_bad_paths)
    val_good_nodes = extract_nodes(val_good_paths)
    val_bad_nodes = extract_nodes(val_bad_paths)

    print("=" * 80)
    print("GEOMETRIC SEPARATION CHECK")
    print("=" * 80)

    print("\n--- Node Statistics ---")
    print(f"Unique nodes in train good paths: {len(train_good_nodes)}")
    print(f"Unique nodes in train bad paths:  {len(train_bad_nodes)}")
    print(f"Unique nodes in val good paths:   {len(val_good_nodes)}")
    print(f"Unique nodes in val bad paths:    {len(val_bad_nodes)}")

    print("\n--- A. EXCLUSIVE NODES (Train Set) ---")

    # Nodes that ONLY appear in good paths (never in bad)
    train_good_only = train_good_nodes - train_bad_nodes
    print(f"Nodes ONLY in GOOD paths (train): {len(train_good_only)}")
    if len(train_good_only) > 0:
        print(f"  -> Model could learn: 'if I see these nodes, it's a good path'")
        print(f"  -> Examples: {list(train_good_only)[:5]}")

    # Nodes that ONLY appear in bad paths (never in good)
    train_bad_only = train_bad_nodes - train_good_nodes
    print(f"Nodes ONLY in BAD paths (train):  {len(train_bad_only)}")
    if len(train_bad_only) > 0:
        print(f"  -> Model could learn: 'if I see these nodes, it's a bad path'")
        print(f"  -> Examples: {list(train_bad_only)[:5]}")

    # Shared nodes
    train_shared = train_good_nodes & train_bad_nodes
    print(f"Nodes in BOTH good and bad (train): {len(train_shared)}")

    print("\n--- B. EXCLUSIVE NODES (Val Set) ---")

    val_good_only = val_good_nodes - val_bad_nodes
    print(f"Nodes ONLY in GOOD paths (val): {len(val_good_only)}")
    if len(val_good_only) > 0:
        print(f"  -> Examples: {list(val_good_only)[:5]}")

    val_bad_only = val_bad_nodes - val_good_nodes
    print(f"Nodes ONLY in BAD paths (val):  {len(val_bad_only)}")
    if len(val_bad_only) > 0:
        print(f"  -> Examples: {list(val_bad_only)[:5]}")

    val_shared = val_good_nodes & val_bad_nodes
    print(f"Nodes in BOTH good and bad (val): {len(val_shared)}")

    print("\n--- C. CROSS-SET NODE OVERLAP ---")

    # Can the model use train-exclusive nodes to classify val?
    val_good_using_train_good_only = len([n for n in val_good_nodes if n in train_good_only])
    val_bad_using_train_bad_only = len([n for n in val_bad_nodes if n in train_bad_only])

    print(f"Val good nodes that were train-good-only: {val_good_using_train_good_only} / {len(val_good_nodes)}")
    print(f"Val bad nodes that were train-bad-only:  {val_bad_using_train_bad_only} / {len(val_bad_nodes)}")

    if val_good_using_train_good_only > 0 or val_bad_using_train_bad_only > 0:
        print(f"  -> 🔥 SHORTCUT DETECTED! Model can use node presence to classify!")

    print("\n--- D. NODE FREQUENCY ANALYSIS ---")

    # Count how often each node appears in good vs bad paths
    from collections import Counter

    train_good_node_counts = Counter()
    for path in train_good_paths:
        for node in path:
            train_good_node_counts[tuple(node)] += 1

    train_bad_node_counts = Counter()
    for path in train_bad_paths:
        for node in path:
            train_bad_node_counts[tuple(node)] += 1

    # Find nodes that are HIGHLY correlated with good or bad
    all_train_nodes = train_good_nodes | train_bad_nodes
    highly_biased_nodes = []

    for node in all_train_nodes:
        good_count = train_good_node_counts[node]
        bad_count = train_bad_node_counts[node]
        total = good_count + bad_count
        if total >= 5:  # Only consider nodes that appear at least 5 times
            good_ratio = good_count / total
            if good_ratio > 0.8 or good_ratio < 0.2:  # Highly biased
                highly_biased_nodes.append((node, good_ratio, total))

    highly_biased_nodes.sort(key=lambda x: abs(x[1] - 0.5), reverse=True)

    print(f"Nodes with >80% bias toward good or bad (min 5 appearances): {len(highly_biased_nodes)}")
    if len(highly_biased_nodes) > 0:
        print("  Top 10 most biased nodes:")
        for node, ratio, count in highly_biased_nodes[:10]:
            label = "GOOD" if ratio > 0.5 else "BAD"
            print(f"    {node}: {ratio*100:.1f}% good ({count} total) -> {label}")
        print(f"  -> ⚠️  Model could learn to recognize these frequently-biased nodes!")

    print("\n--- VERDICT ---")
    if len(train_good_only) > 0 or len(train_bad_only) > 0:
        print("⚠️  WARNING: Exclusive nodes detected - model can use node presence as a shortcut!")
        if len(highly_biased_nodes) > 10:
            print("⚠️  CRITICAL: Many highly-biased nodes - strong geometric separation exists!")
    else:
        print("✓ All nodes appear in both good and bad paths - less obvious geometric shortcuts")
    print("=" * 80)


def classify_with_trivial_features(all_samples_train, all_samples_val):
    """
    Check if the model could use trivial features to classify paths:
    1. Path length
    2. Distance from origin (0,0)
    3. Distance from the closest good node
    4. Distance from the closest bad node
	5. Total Euclidean length
    6. Mean turning angle
    7. Bounding box size
    """
    import math

    def path_length(path):
        return len(path)

    def distance_from_origin(node):
        return math.sqrt(node[0]**2 + node[1]**2)

    def distance_from_closest_node(path, nodes):
        min_dist = float('inf')
        for node in nodes:
            dist = math.sqrt((path[-1][0] - node[0])**2 + (path[-1][1] - node[1])**2)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    # Collect all unique nodes from train and val sets
    def extract_nodes(paths):
        nodes = set()
        for path in paths:
            for node in path:
                nodes.add(tuple(node))
        return nodes
    train_good_nodes = extract_nodes([path for sample in all_samples_train for path in sample["good_paths"]])
    train_bad_nodes = extract_nodes([path for sample in all_samples_train for path in sample["bad_paths"]])
    val_good_nodes = extract_nodes([path for sample in all_samples_val for path in sample["good_paths"]])
    val_bad_nodes = extract_nodes([path for sample in all_samples_val for path in sample["bad_paths"]])
    
    def total_euclidean_length(path):
        total_length = 0
        for i in range(len(path) - 1):
            total_length += math.sqrt((path[i][0] - path[i+1][0])**2 + (path[i][1] - path[i+1][1])**2)
        return total_length

    def mean_turning_angle(path):
        angles = []
        for i in range(1, len(path) - 1):
            v1 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
            v2 = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
            v1_len = math.sqrt(v1[0]**2 + v1[1]**2)
            v2_len = math.sqrt(v2[0]**2 + v2[1]**2)
            # Skip if either vector has zero length
            if v1_len == 0 or v2_len == 0:
                continue
            cos_theta = (v1[0]*v2[0] + v1[1]*v2[1]) / (v1_len * v2_len)
            # Clamp cos_theta to [-1, 1] to avoid numerical errors in acos
            cos_theta = max(-1, min(1, cos_theta))
            angle = math.acos(cos_theta)
            angles.append(angle)
        if angles:
            return sum(angles) / len(angles)
        else:
            return 0
    
    def bounding_box_size(path):
        x_coords = [node[0] for node in path]
        y_coords = [node[1] for node in path]
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        return width * height

    # Calculate features for all paths
    train_good_features = []
    train_bad_features = []
    val_good_features = []
    val_bad_features = []
    
    for sample in all_samples_train:
        for path in sample["good_paths"]:
            train_good_features.append([
                path_length(path),
                distance_from_origin(path[0]),
                distance_from_closest_node(path, train_good_nodes),
                distance_from_closest_node(path, train_bad_nodes),
                total_euclidean_length(path),
                mean_turning_angle(path),
                bounding_box_size(path)
            ])
        for path in sample["bad_paths"]:
            train_bad_features.append([
                path_length(path),
                distance_from_origin(path[0]),
                distance_from_closest_node(path, train_good_nodes),
                distance_from_closest_node(path, train_bad_nodes),
                total_euclidean_length(path),
                mean_turning_angle(path),
                bounding_box_size(path)
            ])
    for sample in all_samples_val:
        for path in sample["good_paths"]:
            val_good_features.append([
                path_length(path),
                distance_from_origin(path[0]),
                distance_from_closest_node(path, val_good_nodes),
                distance_from_closest_node(path, val_bad_nodes),
                total_euclidean_length(path),
                mean_turning_angle(path),
                bounding_box_size(path)
            ])
        for path in sample["bad_paths"]:
            val_bad_features.append([
                path_length(path),
                distance_from_origin(path[0]),
                distance_from_closest_node(path, val_good_nodes),
                distance_from_closest_node(path, val_bad_nodes),
                total_euclidean_length(path),
                mean_turning_angle(path),
                bounding_box_size(path)
            ])
    
    # Convert to numpy arrays for easier manipulation
    train_good_features = np.array(train_good_features)
    train_bad_features = np.array(train_bad_features)
    val_good_features = np.array(val_good_features)
    val_bad_features = np.array(val_bad_features)
    
    # sklearn for classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    clf = RandomForestClassifier()
    clf.fit(np.vstack((train_good_features, train_bad_features)), [1]*len(train_good_features) + [0]*len(train_bad_features))
    
    # Predict and evaluate on val set
    val_features = np.vstack((val_good_features, val_bad_features))
    val_labels = [1]*len(val_good_features) + [0]*len(val_bad_features)
    predictions = clf.predict(val_features)
    accuracy = accuracy_score(val_labels, predictions)
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    # print_stats(all_samples_train, all_samples_val)
    # check_leaks(all_samples_train, all_samples_val)
    # check_separation(all_samples_train, all_samples_val)
    
    classify_with_trivial_features(all_samples_train, all_samples_val)
    
    """
    @neeraja: this is FABULOUS!
    """

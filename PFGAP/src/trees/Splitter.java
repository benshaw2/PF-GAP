package trees;

import java.io.Serializable;
import java.util.*;

import core.AppContext;
//import core.contracts.Dataset;
import core.contracts.ObjectDataset;
//import datasets.ListDataset;
import datasets.ListObjectDataset;
import distance.DistanceMeasure;

/**
 * 
 * @author shifaz
 * @email ahmed.shifaz@monash.edu
 *
 */
// public class Splitter{
public class Splitter implements Serializable {
	
	protected int num_children; //may be move to splitter?
	protected DistanceMeasure distance_measure;
	//protected double[][] exemplars;
	protected Object[] exemplars;
	
	protected DistanceMeasure temp_distance_measure;
	//protected double[][] temp_exemplars;
	protected Object[] temp_exemplars;
	
	//ListDataset[] best_split = null;
	ListObjectDataset[] best_split = null;
	ProximityTree.Node node;
	
	public Splitter(ProximityTree.Node node) throws Exception {
		this.node = node;	
	}
	
	//public ListDataset[] split_data(Dataset sample, Map<Integer, ListDataset> data_per_class) throws Exception {
	public ListObjectDataset[] split_data(
			ObjectDataset sample,
			Map<Object, ListObjectDataset> data_per_class
	) throws Exception {

		ListObjectDataset[] splits;

		if (sample == null || sample.size() == 0) {
			return null;
		}

		if (AppContext.isIsolationMode()) {

			int branches = Math.max(2, AppContext.isolation_num_branches);
			branches = Math.min(branches, sample.size());

			if (branches < 2) {
				return null;
			}

			temp_exemplars = new Object[branches];
			splits = new ListObjectDataset[branches];

			int[] exemplarIndices = sampleDistinctIndices(sample.size(), branches);

			for (int b = 0; b < branches; b++) {
				splits[b] = new ListObjectDataset(sample.size());
				temp_exemplars[b] = sample.get_series(exemplarIndices[b]);
			}

		} else if (AppContext.isRegressionMode()) {

			int branches = Math.max(2, AppContext.regression_num_branches);
			branches = Math.min(branches, sample.size());

			if (branches < 2) {
				return null;
			}

			temp_exemplars = new Object[branches];
			splits = new ListObjectDataset[branches];

			int[] exemplarIndices = sampleDistinctIndices(sample.size(), branches);

			for (int b = 0; b < branches; b++) {
				splits[b] = new ListObjectDataset(sample.size());
				temp_exemplars[b] = sample.get_series(exemplarIndices[b]);
			}

		} else {

			if (data_per_class == null || data_per_class.size() < 2) {
				return null;
			}

			splits = new ListObjectDataset[sample.get_num_classes()];
			temp_exemplars = new Object[sample.get_num_classes()];

			int branch = 0;

			for (Map.Entry<Object, ListObjectDataset> entry : data_per_class.entrySet()) {

				if (entry.getValue() == null || entry.getValue().size() == 0) {
					return null;
				}

				splits[branch] = new ListObjectDataset(sample.size());

				int r = AppContext.getRand().nextInt(entry.getValue().size());
				temp_exemplars[branch] = entry.getValue().get_series(r);

				branch++;
			}
		}

		for (int j = 0; j < sample.size(); j++) {
			int closest_branch = find_closest_branch(
					sample.get_series(j),
					temp_distance_measure,
					temp_exemplars
			);

			splits[closest_branch].add(
					sample.get_class(j),
					sample.get_series(j),
					sample._internal_indices_list().get(j)
			);
		}

		return splits;
	}

	//public int find_closest_branch(double[] query, DistanceMeasure dm, double[][] e) throws Exception{
	public int find_closest_branch(Object query, DistanceMeasure dm, Object[] e) throws Exception{
		return dm.find_closest_node(query, e, true, this.node.tree.getDistance_file());
	}	
	
	//public int find_closest_branch(double[] query) throws Exception{
	public int find_closest_branch(Object query) throws Exception{
		return this.distance_measure.find_closest_node(query, exemplars, true, this.node.tree.getDistance_file());
	}		
	
	//public Dataset[] getBestSplits() {
	public ObjectDataset[] getBestSplits() {
		return this.best_split;
	}
	
	//public ListDataset[] find_best_split(Dataset data) throws Exception {
	//public ListDataset[] find_best_split(ObjectDataset data) throws Exception {
	public ListObjectDataset[] find_best_split(ObjectDataset data) throws Exception {

		// reset each time this method is called:
		best_split = null;
		distance_measure = null;
		exemplars = null;

		if (data == null || data.size() < 2) {
			return null;
		}

		Map<Object, ListObjectDataset> data_per_class =
				(AppContext.isRegressionMode() || AppContext.isIsolationMode())
						? null
						: data.split_classes();

		double best_weighted_purity = Double.POSITIVE_INFINITY;
		ListObjectDataset[] splits = null;
		int parent_size = data.size();

		for (int i = 0; i < AppContext.num_candidates_per_split; i++) {
			temp_distance_measure = selectDistanceMeasure(data);

			splits = (AppContext.isRegressionMode() || AppContext.isIsolationMode())
					? split_data(data, null)
					: split_data(data, data_per_class);

			double weighted_purity = weighted_purity(parent_size, splits);

			// no valid split condition:
			if (Double.isInfinite(weighted_purity) || Double.isNaN(weighted_purity)){
				continue;
			}

			if (weighted_purity < best_weighted_purity) {
				best_weighted_purity = weighted_purity;
				best_split = splits;
				distance_measure = temp_distance_measure;
				exemplars = temp_exemplars;
			}
		}

		if (best_split == null) {
			return null;
			//throw new IllegalStateException(
			//		"No valid split found. Try increasing num_candidates_per_split, "
			//				+ "changing distances, or reducing the number of branches."
			//);
		}

		this.num_children = best_split.length;
		return this.best_split;
	}

	private DistanceMeasure selectDistanceMeasure(ObjectDataset data) throws Exception {
		DistanceMeasure dm;
		if (node.tree.getChosen_distances().length == 0) {
			if (AppContext.random_dm_per_node) {
				int r = AppContext.getRand().nextInt(AppContext.enabled_distance_measures.length);
				dm = new DistanceMeasure(AppContext.enabled_distance_measures[r], AppContext.Descriptors.get(r));
			} else {
				dm = node.tree.tree_distance_measure;
			}
		} else {
			if (AppContext.random_dm_per_node) {
				int r = AppContext.getRand().nextInt(node.tree.getChosen_distances().length);
				dm = new DistanceMeasure(node.tree.getChosen_distances()[r], AppContext.Descriptors.get(r));
			} else {
				dm = node.tree.tree_distance_measure;
			}
		}

		dm.select_random_params(data, AppContext.getRand());
		return dm;
	}


	//public double weighted_gini(int parent_size, ListDataset[] splits) {
	/*public double weighted_purity(int parent_size, ListObjectDataset[] splits) {
		double wpurity = 0.0;
		for (ListObjectDataset split : splits) {
			wpurity += ((double) split.size() / parent_size) * split.purity(AppContext.purity_measure);
		}
		return wpurity;
	}*/

	public double weighted_purity(int parent_size, ListObjectDataset[] splits) {
		return purity.SplitScorer.compute(
				AppContext.purity_measure,
				parent_size,
				splits
		);
	}

	// patch: we should sample exemplars *without* replacement
	private int[] sampleDistinctIndices(int sampleSize, int count) {

		count = Math.min(count, sampleSize);

		List<Integer> indices = new ArrayList<>();

		for (int i = 0; i < sampleSize; i++) {
			indices.add(i);
		}

		Collections.shuffle(indices, AppContext.getRand());

		int[] selected = new int[count];

		for (int i = 0; i < count; i++) {
			selected[i] = indices.get(i);
		}

		return selected;
	}
	
}

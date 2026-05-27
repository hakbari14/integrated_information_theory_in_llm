from sklearn import decomposition
from abc import ABC, abstractmethod
from .enums_class import tpm_creation_type_enum, last_layer_computation_type_enum, granularity_enum, iit_threashold_type_enum
from integrated_information_theory.entity.iit_token_entity import iit_token_entity
from integrated_information_theory.entity.iit_entity import iit_entity
from integrated_information_theory.entity.dimension_entity import dimension_entity
import numpy as np
import pyphi
import math
import sys

class integrated_information_theory(ABC): 

    def __init__(self, config):
        self.seed = 42
        self.set_config(config)
        self.get_config().validate()

    def calculate(self, iit_entity_list): 
        if iit_entity_list is None or len(iit_entity_list) == 0:
            return []
        
        for entity in iit_entity_list:
            try:
                self.refine_embedding(entity)
            except Exception as e:
                print(f"[Error] calculate[refine_embedding]: {e}")

        calcutable_list = list(filter(lambda x: x.has_completion_embedding_for_pca() , iit_entity_list))
        if calcutable_list is None or len(calcutable_list) == 0:
            return []

        if tpm_creation_type_enum.TRAJECTORY == self.get_config().get_tpm_creation_type(): 
            return self.calculate_per_trajectory(calcutable_list)
        elif tpm_creation_type_enum.PROMPT == self.get_config().get_tpm_creation_type():
            return self.calculate_per_prompt(calcutable_list)
        elif tpm_creation_type_enum.BATCH == self.get_config().get_tpm_creation_type():
            return self.calculate_iit_reward_mean_std(calcutable_list)
        else:
            return []

    def calculate_prompt(self, iit_entity_list): 
        if iit_entity_list is None or len(iit_entity_list) == 0:
            return None
        
        for entity in iit_entity_list:
            try:
                self.refine_embedding(entity)
            except Exception as e:
                print(f"[Error] calculate[refine_embedding]: {e}")

        calcutable_list = list(filter(lambda x: x.has_completion_embedding_for_pca() , iit_entity_list))
        if calcutable_list is None or len(calcutable_list) == 0:
            return None

        return self.calculate_iit_reward_prompt(calcutable_list)

    def calculate_entity(self, entity): 
        if entity is None or not entity.is_calcutable():
            return entity
        
        try:
            self.refine_embedding(entity)
        except Exception as e:
            print(f"[Error] calculate[refine_embedding]: {e}")
            return entity

        single_iit_entity_list = []
        single_iit_entity_list.append(entity)
        single_iit_entity_list = self.calculate_iit_reward_mean_std(single_iit_entity_list)
        return single_iit_entity_list[0]

    def calculate_per_trajectory(self, iit_entity_list): 
        new_iit_entity_list = []
        for entity in iit_entity_list:
            single_iit_entity_list = []
            single_iit_entity_list.append(entity)
            single_iit_entity_list = self.calculate_iit_reward_mean_std(single_iit_entity_list)
            new_iit_entity_list.extend(single_iit_entity_list)

        return new_iit_entity_list

    def calculate_per_prompt(self, iit_entity_list): 
        prompt_set = set(map(lambda x: x.get_promptID(), iit_entity_list))
        new_iit_entity_list = []
        for prompt_ID in prompt_set:
            prompt_iit_entity_list = list(filter(lambda x: x.get_promptID() == prompt_ID, iit_entity_list))
            if len(prompt_iit_entity_list) == 0:
                continue

            new_prompt_iit_entity_list = self.calculate_iit_reward_mean_std(prompt_iit_entity_list)
            new_iit_entity_list.extend(new_prompt_iit_entity_list)

        return new_iit_entity_list

    def calculate_iit_reward_mean_std(self, iit_entity_list):
        if len(iit_entity_list) == 0:
            return iit_entity_list
        
        start_dimension, end_dimension = self.calculate_reduced_dimention(self.calculate_tokens_count(iit_entity_list))
        selection_dimension_list = []        
        for dimension in range(start_dimension , end_dimension + 1):
            iit_entity_list_copy = iit_entity.clone_list(iit_entity_list)
            tokens_count = self.calculate_tokens_count(iit_entity_list_copy)
            
            d_entity = dimension_entity(dimension, iit_entity_list_copy)
            selection_dimension_list.append(d_entity)
            for entity in iit_entity_list_copy: 
                try:
                    entity.set_token_count_for_reduced_dim(tokens_count)
                    entity.set_reduced_dim(dimension)
                    entity.set_completion_concatenated_embedding(self.reduce_embedding(entity))
                except Exception as e:
                    print(f"[Error] calculate[reduce_embedding]: {e}")
                    entity.set_completion_concatenated_embedding(None)

            filtered_iit_entity_list_copy = list(filter(lambda x: x.has_concatenated_embedding(), iit_entity_list_copy))
            if len(filtered_iit_entity_list_copy) == 0:
                d_entity.set_mean_standard_deviation([])
                continue

            tpm_sbs = self.build_tpm_list(filtered_iit_entity_list_copy, dimension)
            iit_reward_list = []
            for entity in filtered_iit_entity_list_copy:
                try:
                    weights_ts = self.build_weights(entity)
                    raw_value = self.calculate_iit(entity, tpm_sbs, weights_ts)
                    entity.set_iit_reward_raw(raw_value)
                    iit_value = self.compute_last_layer_on_reward(raw_value)
                    entity.set_iit_reward(iit_value)
                    tpm_loss, tpm_entropy = self.calculate_tpm_loss_entropy(entity, tpm_sbs)
                    entity.set_tpm_loss(tpm_loss)
                    entity.set_tpm_entropy(tpm_entropy)
                except Exception as e:
                    print(f"[Error] calculate[calculate_iit]: {e}")
                    entity.set_iit_reward(0.0)
                    entity.set_iit_reward_raw(0.0)
                    entity.set_iit_reward_raw_actual(0.0)
                
                iit_reward_list.append(entity.get_iit_reward())
            
            d_entity.set_mean_standard_deviation(iit_reward_list)
            iit_entity.release_memory(iit_entity_list_copy)
            
            
        selection_dimension_list = dimension_entity.z_score_normalization(selection_dimension_list)
        max_metric = -sys.float_info.max
        selected_dimension = 3
        best_iit_entity_list = iit_entity.clone_list(iit_entity_list)
        print()
        print(f'W_mean = {self.get_config().get_coefficient_mean_reward_dimension()}, W_std = {self.get_config().get_coefficient_std_reward_dimension()}')
        for d_entity in selection_dimension_list:
            metric = d_entity.get_difference_metric(self.get_config())
            print(f'dimension = {d_entity.get_dimension()}, mean = {d_entity.get_normalized_mean_iit_reward()}, std = {d_entity.get_normalized_std_iit_reward()}, diff = {metric}, reward_list = {d_entity.get_iit_reward_list()}')
            if metric > max_metric:
                max_metric = metric
                selected_dimension = d_entity.get_dimension()
                best_iit_entity_list = d_entity.get_iit_list()
        print()

        print(f'selected_dimension = {selected_dimension}, max_metric = {max_metric}')
        iit_entity.release_memory(iit_entity_list)
        iit_entity.release_memory(best_iit_entity_list)
        return best_iit_entity_list

    def calculate_iit_reward_prompt(self, iit_entity_list):
        if len(iit_entity_list) == 0:
            return None

        dimension = self.get_config().get_reduced_dim()        
        tokens_count = self.calculate_tokens_count(iit_entity_list)
        for entity_tpm in iit_entity_list: 
            try:
                entity_tpm.set_token_count_for_reduced_dim(tokens_count)
                entity_tpm.set_reduced_dim(dimension)
                entity_tpm.set_completion_concatenated_embedding(self.reduce_embedding(entity_tpm))
            except Exception as e:
                print(f"[Error] calculate[reduce_embedding]: {e}")
                entity_tpm.set_completion_concatenated_embedding(None)

        filtered_iit_entity_list_copy = list(filter(lambda x: x.has_concatenated_embedding(), iit_entity_list))
        if len(filtered_iit_entity_list_copy) == 0:
            return None
        
        tpm_sbs = self.build_tpm_list(filtered_iit_entity_list_copy, dimension)
        for entity in filtered_iit_entity_list_copy:
            try:
                weights_ts = self.build_weights(entity)
                raw_value = self.calculate_iit(entity, tpm_sbs, weights_ts)
                entity.set_iit_reward_raw(raw_value)
                iit_value = self.compute_last_layer_on_reward(raw_value)
                entity.set_iit_reward(iit_value)
                tpm_loss, tpm_entropy = self.calculate_tpm_loss_entropy(entity, tpm_sbs)
                entity.set_tpm_loss(tpm_loss)
                entity.set_tpm_entropy(tpm_entropy)
            except Exception as e:
                print(f"[Error] calculate[calculate_iit]: {e}")
                entity.set_iit_reward(0.0)
                entity.set_iit_reward_raw(0.0)
                entity.set_iit_reward_raw_actual(0.0)
        
        entity = min(filtered_iit_entity_list_copy, key=lambda x: x.get_tpm_entropy())
        iit_entity.release_memory(iit_entity_list)
        return entity

    def calculate_reduced_dimention(self, tokens_count):
        if self.get_config().get_adaptive_dim() == False:
            return self.get_config().get_reduced_dim(), self.get_config().get_reduced_dim()
        
        for i in range(3,10):
            next_tpm_matrix_size = pow(2, i + 1) * pow(2, i + 1)
            if next_tpm_matrix_size <= tokens_count: 
                continue
            return 3, i
            
        return 3, 10

    def calculate_tokens_count(self, iit_entity_list):
        tokens_count = 0
        for entity in iit_entity_list:
            if entity.get_completion_embedding_for_pca() is None: 
                continue

            completion_token_count = entity.get_completion_embedding_for_pca().shape[0]
            tokens_count += completion_token_count 

        return tokens_count

    @abstractmethod
    def calculate_iit(tpm_sbs, weights_ts):
        pass

    def calculate_tpm_loss_entropy(self, entity, tpm_sbs):
        markov_chain = entity.get_markov_chain()
        tpm_loss, tpm_entropy = 0, 0
        for s1, s2 in zip(markov_chain, markov_chain[1:]):
            current_state = pyphi.convert.state2le_index(s1)
            next_state = pyphi.convert.state2le_index(s2)
            tpm_loss += -1 * math.log(tpm_sbs[current_state,next_state])
            tpm_entropy += -1 * tpm_sbs[current_state,next_state] * math.log(tpm_sbs[current_state,next_state])
        
        tpm_loss /= (len(markov_chain) -1)
        tpm_entropy /= (len(markov_chain) -1)
        return tpm_loss, tpm_entropy

    def refine_embedding(self, entity): 
        prompt_embedding = entity.get_prompt_embedding()
        response_embedding = entity.get_completion_embedding()
        attention_score = self.attention_score(prompt_embedding, response_embedding)
        attention_weight = self.attention_weight(attention_score)
        attended_response = np.matmul(attention_weight, prompt_embedding)
        for t_entity in entity.get_iit_token_list():
            t_entity.set_attended_embedding(attended_response[:, t_entity.get_token_number(), :])

        response_concatenated = np.concatenate(tuple(attended_response), axis=1,)
        response_for_pca = response_concatenated
        if granularity_enum.CHUNK == self.get_config().get_granularity(): 
            number_of_subarray = math.ceil(response_concatenated.shape[0] / self.get_config().get_chunk_size())

            chunk_list = []
            ii_entity_token_list = []
            max_dim = 0
            start = 0
            for idx, chunk in enumerate(np.array_split(response_concatenated, number_of_subarray)): 
                c = np.concatenate(tuple(chunk), axis=0,)
                max_dim = max(max_dim, c.shape[0])
                chunk_list.append(c)

                token, emb = entity.aggregate_token_list(start, len(chunk))
                start += len(chunk)
                t_entity = iit_token_entity(idx, token)
                t_entity.set_attended_embedding(emb)
                ii_entity_token_list.append(t_entity)


            refined_chunk_list = []
            for chunk in chunk_list: 
                if chunk.shape[0] == max_dim: 
                    refined_chunk_list.append(chunk)
                    continue
                new_chunk = np.pad(chunk, (0, max_dim - chunk.shape[0]), 'constant', constant_values = 0)
                refined_chunk_list.append(new_chunk)
            
            entity.set_iit_token_list(ii_entity_token_list)
            response_for_pca = np.array(refined_chunk_list, dtype=float)
            
        entity.set_completion_embedding_for_pca(response_for_pca)
        return entity

    def reduce_embedding(self, entity): 
        if entity.get_completion_embedding_for_pca() is None:
            raise Exception(f'Completion Embedding is Null')
        
        if entity.get_completion_embedding_for_pca().shape[0] < entity.get_reduced_dim():
            raise Exception(f'Embedding with {entity.get_completion_embedding_for_pca().shape} does not have the capability for dimensionality reduction to {entity.get_reduced_dim()}')
        
        pca = decomposition.PCA(n_components = entity.get_reduced_dim(), random_state = self.seed + 42 * 5,)
        pca.fit(entity.get_completion_embedding_for_pca())
        return pca.transform(entity.get_completion_embedding_for_pca())

    def compute_last_layer_on_reward(self, x):
        if last_layer_computation_type_enum.EXP == self.get_config().get_last_layer_computation_type(): 
            return 1 - math.exp(-1 * self.get_config().get_last_layer_computation_param() * x)

        if last_layer_computation_type_enum.TANH == self.get_config().get_last_layer_computation_type(): 
            return math.tanh(self.get_config().get_last_layer_computation_param() * x)

        if last_layer_computation_type_enum.IDENTITY == self.get_config().get_last_layer_computation_type(): 
            return x
        
        return 0.0

    def attention_score(self, prompt, response, l_mask_spans=None, l_mask_context=None):
        """
        Calculate attention score between prompt and response tensors.

        Args:
        - prompt: numpy array of shape (n_l, n_1, D) - prompt embeddings
        - response: numpy array of shape (n_l, n_2, D) - response embeddings

        Returns:
        - attention_scores: numpy array of shape (n_l, n_2, n_1) - attention scores
        """
        # Get dimensions
        n_l, n_1, D = prompt.shape
        _, n_2, _ = response.shape

        # Calculate the dot product between response and prompt embeddings
        # The resulting shape will be (n_l, n_2, n_1)
        attention_scores = np.matmul(response, prompt.transpose(0, 2, 1))  # (n_l, n_2, n_1)

        # Scale the attention scores by sqrt(D)
        attention_scores = attention_scores / np.sqrt(D)

        if (l_mask_spans is not None) and (l_mask_context is not None):

            # Create a boolean mask for indices in l_mask_context
            mask_context = np.zeros(n_1, dtype=bool)
            mask_context[l_mask_context] = True  # Set context indices to True

            # Create a boolean mask for indices in l_mask_spans
            mask_span = np.zeros(n_1, dtype=bool)
            mask_span[l_mask_spans] = True  # Set span indices to True

            # Create a mask for indices that are neither in l_mask_spans nor in l_mask_context
            mask_out_of_both = ~mask_span & ~mask_context

            # Modify attention scores where the index is in l_mask_context
            attention_scores[:, :, mask_context] *= 0.6

            # Modify attention scores where the index is not in l_mask_spans or l_mask_context
            attention_scores[:, :, mask_out_of_both] *= 0.2

        # attention_scores = softmax(attention_scores, axis=-1)  # Softmax over the last axis (n_1 dimension)

        return attention_scores


    # Apply softmax to get the final attention scores for each layer
    def attention_weight(self, attention_scores, axis=-1):
        # Subtract the max value of x along the specified axis to prevent overflow
        exps = np.exp(attention_scores - np.max(attention_scores, axis=axis, keepdims=True))
        # Calculate the sum of the exponentials along the specified axis
        # Divide each exponential value by the sum of exponentials to get probabilities
        return exps / np.sum(exps, axis=axis, keepdims=True)

    def build_tpm(self, time_series):
        # This function is used to construct a state-by-state TPM.
        #
        # Inputs:
        # 1) time_series; array with dimensions n_time_points X n_regions
        #
        # Outputs:
        # 1) tpm:       array with dimensions n_states X n_states)
        #               each entry gives the sum_probabilities of transition between two states.
        #
        # 2) weights:   array with dimensions 1 X n_states
        #               each entry gives the sum_probabilities of a state appearing in the time-series.

        # Obtain binary time-series based on mean signal threshold.
        avgs = np.mean(time_series, axis=0)
        time_series_copy = np.copy(time_series)

        for i in range(len(avgs)):
            time_series[np.where(time_series_copy[:, i] >= avgs[i]), i] = 1
            time_series[np.where(time_series_copy[:, i] < avgs[i]), i] = 0

        time_series = time_series.astype(int)

        markov_chain = time_series.tolist()
        n = len(markov_chain[0])
        tpm = np.zeros((2**n, 2**n))

        # Loop through all transitions and populate TPM.
        for s1, s2 in zip(markov_chain, markov_chain[1:]):
            i = pyphi.convert.state2le_index(s1)
            j = pyphi.convert.state2le_index(s2)
            tpm[i][j] += 1

        # Create array for transition counts.
        transitions_total = np.sum(tpm, axis=-1)

        # Normalize counts in TPM to obtain probabilities.
        for div in range(len(transitions_total)):
            if transitions_total[div] != 0.0:
                tpm[div, :] /= transitions_total[div]

        # Create array for state counts.
        weights_ts = np.zeros((2 ** time_series.shape[-1]))

        for s in markov_chain:
            i = pyphi.convert.state2le_index(s)
            weights_ts[i] += 1

        # Normalize weights with respect to time-series length:
        weights_ts /= len(markov_chain)
        return markov_chain, np.copy(tpm), np.copy(weights_ts)

    
    def build_tpm_list(self, iit_entity_list, reduced_dim):

        # This function is used to construct a state-by-state TPM.
        #
        # Inputs:
        # 1) list of time_series; array with dimensions n_time_points X n_regions
        #
        # Outputs:
        # 1) tpm:       array with dimensions n_states X n_states)
        #               each entry gives the sum_probabilities of transition between two states.

        # Obtain binary time-series based on mean signal threshold.
        time_series_list = list(map(lambda x: x.get_completion_concatenated_embedding(), iit_entity_list))
        concatenated_time_series = np.concatenate(time_series_list, axis=0)
        
        if iit_threashold_type_enum.AVERAGE == self.get_config().get_threashold_type(): 
            avgs = np.mean(concatenated_time_series, axis=0)
        elif iit_threashold_type_enum.MEDIAN == self.get_config().get_threashold_type(): 
            avgs = np.median(concatenated_time_series, axis=0)
        
        tpm = np.zeros((2**reduced_dim, 2**reduced_dim))
        for entity in iit_entity_list:
            time_series = entity.get_completion_concatenated_embedding()
            time_series_copy = np.copy(time_series)

            for i in range(len(avgs)):
                time_series[np.where(time_series_copy[:, i] >= avgs[i]), i] = 1
                time_series[np.where(time_series_copy[:, i] < avgs[i]), i] = 0

            time_series = time_series.astype(int)
            markov_chain = time_series.tolist()
            entity.set_all_markov_chain(markov_chain)

            # Loop through all transitions and populate TPM.
            for s1, s2 in zip(markov_chain, markov_chain[1:]):
                i = pyphi.convert.state2le_index(s1)
                j = pyphi.convert.state2le_index(s2)
                tpm[i][j] += 1

        # Create array for transition counts.
        transitions_total = np.sum(tpm, axis=-1)

        # Normalize counts in TPM to obtain probabilities.
        for div in range(len(transitions_total)):
            if transitions_total[div] != 0.0:
                tpm[div, :] /= transitions_total[div]

        return np.copy(tpm)

    def build_weights(self, entity):
        markov_chain = entity.get_markov_chain()
        weights_ts = np.zeros((2 ** entity.get_reduced_dim()))

        for s in markov_chain:
            i = pyphi.convert.state2le_index(s)
            weights_ts[i] += 1

        # Normalize weights with respect to time-series length:
        weights_ts /= len(markov_chain)
        return weights_ts

    def build_markov_chain_list(self, iit_entity_list):
        time_series_list = list(map(lambda x: x.get_completion_concatenated_embedding(), iit_entity_list))
        concatenated_time_series = np.concatenate(time_series_list, axis=0)
        avgs = np.mean(concatenated_time_series, axis=0)
        
        for entity in iit_entity_list:
            time_series = entity.get_completion_concatenated_embedding()
            time_series_copy = np.copy(time_series)

            for i in range(len(avgs)):
                time_series[np.where(time_series_copy[:, i] >= avgs[i]), i] = 1
                time_series[np.where(time_series_copy[:, i] < avgs[i]), i] = 0

            time_series = time_series.astype(int)
            markov_chain = time_series.tolist()
            entity.set_all_markov_chain(markov_chain)


        return iit_entity_list

    def get_config(self): 
        return self.config

    def set_config(self, value): 
        self.config = value



class iit_token_entity:
    
    def __init__(self, token_number, token):
        self.token_number = token_number
        self.token = token
        self.attended_embedding = None
        self.token_markov_chain_0_1_emb = None
        self.state_index = None
        self.iit_cause_value = None
        self.iit_effect_value = None
        self.iit_value = None
        self.iit_cause_state_index = None
        self.iit_effect_state_index = None

    def get_token_number(self):
        return self.token_number

    def set_token_number(self, value):
        self.token_number = value

    def get_token(self):
        return self.token

    def set_token(self, value):
        self.token = value

    def get_attended_embedding(self):
        return self.attended_embedding

    def set_attended_embedding(self, value):
        self.attended_embedding = value

    def get_token_markov_chain_0_1_emb(self):
        return self.token_markov_chain_0_1_emb

    def set_token_markov_chain_0_1_emb(self, value):
        self.token_markov_chain_0_1_emb = value

    def get_state_index(self):
        return self.state_index

    def set_state_index(self, value):
        self.state_index = value

    def get_iit_cause_value(self):
        return self.iit_cause_value

    def set_iit_cause_value(self, value):
        self.iit_cause_value = value

    def get_iit_effect_value(self):
        return self.iit_effect_value

    def set_iit_effect_value(self, value):
        self.iit_effect_value = value

    def get_iit_value(self):
        return self.iit_value

    def set_iit_value(self, value):
        self.iit_value = value

    def get_iit_cause_state_index(self):
        return self.iit_cause_state_index

    def set_iit_cause_state_index(self, value):
        self.iit_cause_state_index = value

    def get_iit_effect_state_index(self):
        return self.iit_effect_state_index

    def set_iit_effect_state_index(self, value):
        self.iit_effect_state_index = value

    @staticmethod
    def clone(token_entity): 
        new_token_entity = iit_token_entity(token_entity.get_token_number(), token_entity.get_token())
        new_token_entity.set_attended_embedding(token_entity.get_attended_embedding())
        new_token_entity.set_token_markov_chain_0_1_emb(token_entity.get_token_markov_chain_0_1_emb())
        new_token_entity.set_state_index(token_entity.get_state_index())
        new_token_entity.set_iit_cause_value(token_entity.get_iit_cause_value())
        new_token_entity.set_iit_effect_value(token_entity.get_iit_effect_value())
        new_token_entity.set_iit_value(token_entity.get_iit_value())
        new_token_entity.set_iit_cause_state_index(token_entity.get_iit_cause_state_index())
        new_token_entity.set_iit_effect_state_index(token_entity.get_iit_effect_state_index())
   
        return new_token_entity

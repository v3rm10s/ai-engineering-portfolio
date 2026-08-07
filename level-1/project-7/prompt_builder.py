class SystemPromptBuilder:
    
    def __init__(self, base_role: str):
        self.base_role = base_role
        self.context = {}
        self.rules = []
    
    def add_context(self, key: str, value: str):
        self.context[key] = value
    
    def add_rule(self, rule: str):
        self.rules.append(rule)
    
    def build(self) -> str:
        context_str = "\n".join(
            f"- {k}: {v}" for k, v in self.context.items()
        )
        rules_str = "\n".join(f"- {r}" for r in self.rules)
        
        return (
            f"### ROLE\n{self.base_role}\n\n"
            f"### CONTEXT\n{context_str}\n\n"
            f"### RULES & CONSTRAINTS\n{rules_str}"
        )
    
builder = SystemPromptBuilder("Expert Technical Tutor")

builder.add_context("User Skill Level", "Beginner")
builder.add_context("Preferred Pace", "Extra Slow")

builder.add_rule("Explain concepts using real-world analogies")
builder.add_rule("Break code snippers down into small, digestible chunks.")

print(builder.build())
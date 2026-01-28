# Test file for super().__init__() through Jac compilation pipeline
# This tests that Python inheritance with super() works correctly
# when routed through Jac's IR and back to Python AST


from super_init_base import Child, GrandChild

if __name__ == "__main__":
    c = Child()
    gc = GrandChild()

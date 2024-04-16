from metaflow import FlowSpec, step

class LinearFlow(FlowSpec):
    @step
    def start(self):
        print("flow starting")
        self.next(self.data)
    @step
    def data(self):
        import data
        data.main()
        self.next(self.model)
    @step
    def model(self):
        import createmodel
        createmodel.main(False)
        self.next(self.end)
    @step
    def end(self):
        print("flow over")

if __name__ == '__main__':
    LinearFlow()
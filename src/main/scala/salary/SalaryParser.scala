package salary

import salary.data.TestData
import salary.data.TrainData
import salary.db.DatabaseManager

object SalaryParser {

  def main(args: Array[String]) = {
    val manager = new DatabaseManager()
    val train = manager.salaryDB("train")
    val test = manager.salaryDB("test")
    TrainData.parseFile("data/Train_rev1.csv", train)
    TestData.parseFile("data/Valid_rev1.csv", test)
  }

}

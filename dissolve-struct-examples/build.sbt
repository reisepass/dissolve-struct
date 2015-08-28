name := "DissolveStructExample"

organization := "ch.ethz.dalab"

version := "0.1-SNAPSHOT"

scalaVersion := "2.10.5"

libraryDependencies += "ch.ethz.dalab" %% "dissolvestruct" % "0.1-SNAPSHOT"

libraryDependencies += "org.scala-lang.modules" %% "scala-pickling" % "0.10.0"

libraryDependencies += "org.scalatest" % "scalatest_2.10" % "2.0" % "test"

libraryDependencies += "org.scalanlp" %% "breeze" % "0.11.2"

libraryDependencies += "gov.nih.imagej" % "imagej" % "1.46"

libraryDependencies += "org.parboiled" %% "parboiled" % "2.1.0"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.4.1"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.4.1"

resolvers += "IESL Release" at "http://dev-iesl.cs.umass.edu/nexus/content/groups/public"

libraryDependencies += "cc.factorie" % "factorie" % "1.0"

libraryDependencies += "com.github.scopt" %% "scopt" % "3.3.0"

resolvers += Resolver.sonatypeRepo("public")

EclipseKeys.createSrc := EclipseCreateSrc.Default + EclipseCreateSrc.Resource


mergeStrategy in assembly <<= (mergeStrategy in assembly) { (old) =>
    {
        case PathList("javax", "servlet", xs @ _*)           => MergeStrategy.first
        case PathList(ps @ _*) if ps.last endsWith ".html"   => MergeStrategy.first
        case "application.conf"                              => MergeStrategy.concat
        case "reference.conf"                                => MergeStrategy.concat
        case "log4j.properties"                              => MergeStrategy.discard
        case m if m.toLowerCase.endsWith("manifest.mf")      => MergeStrategy.discard
        case m if m.toLowerCase.matches("meta-inf.*\\.sf$")  => MergeStrategy.discard
        case _ => MergeStrategy.first
    }
}


test in assembly := {}
